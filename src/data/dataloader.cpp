#include "data/dataloader.h"
#include "execution/async_io.h"
#include "index/range_posting_index.h"
#include "data/fp16_l2.h"
#include "planning/query_prepare_parallel.h"
#include "predicates/range_filter.cuh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <fcntl.h>
#include <sstream>
#include <atomic>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace {

std::mutex query_preparation_config_mutex;
QueryPreparationConfig query_preparation_config;
QueryPreparationConfig snapshot_query_preparation_config() {
    std::lock_guard<std::mutex> lock(query_preparation_config_mutex);
    return query_preparation_config;
}

using error = std::runtime_error;
using json = nlohmann::json;
constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
[[noreturn]] void fail_graph(const std::string& message)
{
    throw error("graph runtime setup failed: " + message);
}

void note_full_base_loader_call(const char* loader_name, const std::string& path)
{
    const char* fail_path = std::getenv("GO_FANN_FAIL_ON_FULL_BASE_LOAD_PATH");
    if (fail_path == nullptr || path != fail_path) {
        return;
    }
    fail_graph(std::string(loader_name) + " unexpectedly attempted to load full base vector payload from " + path);
}

float half_to_float(uint16_t value) noexcept
{
    return go_fann::fp16_l2::decode(value);
}

void convert_half_row_to_float(const uint16_t* src, float* dst, uint32_t cols)
{
    for (uint32_t col = 0; col < cols; ++col) {
        dst[col] = half_to_float(src[col]);
    }
}

void read_exact_at(int fd, void* dst, size_t bytes, uint64_t offset, const std::string& description)
{
    uint8_t* out = static_cast<uint8_t*>(dst);
    size_t done = 0;
    while (done < bytes) {
        const ssize_t nread = pread(fd,
                                    out + done,
                                    bytes - done,
                                    static_cast<off_t>(offset + done));
        if (nread <= 0) {
            fail_graph("failed to read " + description);
        }
        done += static_cast<size_t>(nread);
    }
}

struct GroupPostingData {
    std::vector<std::vector<uint32_t>> sample_ids_by_group;
    std::vector<std::vector<uint64_t>> filter_masks_by_group;
};

struct SpmatHeaderInfo {
    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t nnz = 0;
    uint32_t effective_rows = 0;
};

uint64_t fnv1a_mix_u64(uint64_t hash, uint64_t value)
{
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & 0xffULL;
        hash *= kFnvPrime;
    }
    return hash;
}

using LoadClock = std::chrono::steady_clock;

bool graph_load_timing_enabled()
{
    static const bool enabled =
        std::getenv("GO_FANN_LOAD_TIMING") != nullptr ||
        std::getenv("GOFANN_DEBUG_RUNTIME") != nullptr;
    return enabled;
}

void log_graph_load_stage(const char* stage, LoadClock::time_point start)
{
    if (!graph_load_timing_enabled()) {
        return;
    }
    const auto elapsed = std::chrono::duration<double, std::milli>(LoadClock::now() - start).count();
    std::fprintf(stdout, "[runtime-load-time] stage=%s ms=%.3f\n", stage, elapsed);
    std::fflush(stdout);
}

void log_graph_build_stage(const char* stage, LoadClock::time_point start)
{
    const auto elapsed = std::chrono::duration<double, std::milli>(LoadClock::now() - start).count();
    std::fprintf(stdout, "[graph-build-prep] stage=%s ms=%.3f\n", stage, elapsed);
    std::fflush(stdout);
}

uint64_t elapsed_load_ns(LoadClock::time_point start)
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(LoadClock::now() - start).count());
}

double ns_to_ms(uint64_t ns)
{
    return static_cast<double>(ns) / 1000000.0;
}

struct QueryManifestLoadProfile {
    uint64_t total_ns = 0;
    uint64_t query_work_wall_ns = 0;
    uint64_t discover_ngram_ns = 0;
    uint64_t global_ngram_load_ns = 0;
    uint64_t operator_index_load_ns = 0;
    uint64_t file_read_ns = 0;
    uint64_t fallback_file_read_ns = 0;
    uint64_t operator_lookup_ns = 0;
    uint64_t operator_parity_check_ns = 0;
    uint64_t fallback_index_json_parse_ns = 0;
    uint64_t fallback_index_add_token_ns = 0;
    uint64_t fallback_index_finalize_ns = 0;
    uint64_t json_parse_ns = 0;
    uint64_t record_filter_ns = 0;
    uint64_t family_subtype_ns = 0;
    uint64_t exact_extract_ns = 0;
    uint64_t fuzzy_expand_ns = 0;
    uint64_t fuzzy_expand_wall_ns = 0;
    uint64_t fuzzy_row_finalize_ns = 0;
    uint64_t append_row_ns = 0;
    uint64_t expand_normalize_ngram_ns = 0;
    uint64_t expand_posting_lookup_ns = 0;
    uint64_t expand_score_filter_ns = 0;
    uint64_t expand_candidate_sort_ns = 0;
    uint64_t expand_label_finalize_ns = 0;

    uint64_t empty_lines = 0;
    uint64_t parsed_records = 0;
    uint64_t skipped_records = 0;
    uint64_t fuzzy_predicates = 0;
    uint64_t fuzzy_patterns = 0;
    uint64_t pattern_chars = 0;
    uint64_t grams_total = 0;
    uint64_t grams_found = 0;
    uint64_t grams_missing = 0;
    uint64_t postings_scanned = 0;
    uint64_t overlap_candidates = 0;
    uint64_t scored_candidates = 0;
    uint64_t emitted_labels = 0;
    uint64_t operator_lookups = 0;
    uint64_t operator_direct_only_hits = 0;
    uint64_t operator_fallbacks = 0;
    uint64_t operator_direct_candidates = 0;
    uint64_t operator_direct_matches = 0;
    uint64_t operator_baseline_grams = 0;
    uint64_t operator_baseline_postings = 0;
    uint64_t operator_baseline_candidates = 0;
    uint64_t operator_parity_rows = 0;
    uint64_t final_row_labels = 0;
    uint64_t output_nnz = 0;
    uint64_t max_grams = 0;
    uint64_t max_postings = 0;
    uint64_t max_overlap_candidates = 0;
    uint64_t max_scored_candidates = 0;
    uint64_t max_emitted_labels = 0;
    uint64_t max_final_row_labels = 0;
    uint64_t max_expand_ns = 0;
    uint64_t fuzzy_expand_threads = 0;
};

thread_local QueryManifestLoadProfile* active_query_manifest_profile = nullptr;

struct ScopedQueryManifestProfile {
    explicit ScopedQueryManifestProfile(QueryManifestLoadProfile* profile)
        : previous(active_query_manifest_profile)
    {
        active_query_manifest_profile = profile;
    }

    ~ScopedQueryManifestProfile()
    {
        active_query_manifest_profile = previous;
    }

    QueryManifestLoadProfile* previous = nullptr;
};

void merge_query_manifest_profile(QueryManifestLoadProfile& dst, const QueryManifestLoadProfile& src)
{
    dst.discover_ngram_ns += src.discover_ngram_ns;
    dst.global_ngram_load_ns += src.global_ngram_load_ns;
    dst.operator_index_load_ns += src.operator_index_load_ns;
    dst.operator_lookup_ns += src.operator_lookup_ns;
    dst.operator_parity_check_ns += src.operator_parity_check_ns;
    dst.fallback_index_json_parse_ns += src.fallback_index_json_parse_ns;
    dst.fallback_index_add_token_ns += src.fallback_index_add_token_ns;
    dst.fallback_index_finalize_ns += src.fallback_index_finalize_ns;
    dst.json_parse_ns += src.json_parse_ns;
    dst.record_filter_ns += src.record_filter_ns;
    dst.family_subtype_ns += src.family_subtype_ns;
    dst.exact_extract_ns += src.exact_extract_ns;
    dst.fuzzy_expand_ns += src.fuzzy_expand_ns;
    dst.fuzzy_expand_wall_ns += src.fuzzy_expand_wall_ns;
    dst.fuzzy_row_finalize_ns += src.fuzzy_row_finalize_ns;
    dst.append_row_ns += src.append_row_ns;
    dst.expand_normalize_ngram_ns += src.expand_normalize_ngram_ns;
    dst.expand_posting_lookup_ns += src.expand_posting_lookup_ns;
    dst.expand_score_filter_ns += src.expand_score_filter_ns;
    dst.expand_candidate_sort_ns += src.expand_candidate_sort_ns;
    dst.expand_label_finalize_ns += src.expand_label_finalize_ns;

    dst.empty_lines += src.empty_lines;
    dst.parsed_records += src.parsed_records;
    dst.skipped_records += src.skipped_records;
    dst.fuzzy_predicates += src.fuzzy_predicates;
    dst.fuzzy_patterns += src.fuzzy_patterns;
    dst.pattern_chars += src.pattern_chars;
    dst.grams_total += src.grams_total;
    dst.grams_found += src.grams_found;
    dst.grams_missing += src.grams_missing;
    dst.postings_scanned += src.postings_scanned;
    dst.overlap_candidates += src.overlap_candidates;
    dst.scored_candidates += src.scored_candidates;
    dst.emitted_labels += src.emitted_labels;
    dst.operator_lookups += src.operator_lookups;
    dst.operator_direct_only_hits += src.operator_direct_only_hits;
    dst.operator_fallbacks += src.operator_fallbacks;
    dst.operator_direct_candidates += src.operator_direct_candidates;
    dst.operator_direct_matches += src.operator_direct_matches;
    dst.operator_baseline_grams += src.operator_baseline_grams;
    dst.operator_baseline_postings += src.operator_baseline_postings;
    dst.operator_baseline_candidates += src.operator_baseline_candidates;
    dst.operator_parity_rows += src.operator_parity_rows;
    dst.final_row_labels += src.final_row_labels;
    dst.output_nnz += src.output_nnz;

    dst.max_grams = std::max(dst.max_grams, src.max_grams);
    dst.max_postings = std::max(dst.max_postings, src.max_postings);
    dst.max_overlap_candidates = std::max(dst.max_overlap_candidates, src.max_overlap_candidates);
    dst.max_scored_candidates = std::max(dst.max_scored_candidates, src.max_scored_candidates);
    dst.max_emitted_labels = std::max(dst.max_emitted_labels, src.max_emitted_labels);
    dst.max_final_row_labels = std::max(dst.max_final_row_labels, src.max_final_row_labels);
    dst.max_expand_ns = std::max(dst.max_expand_ns, src.max_expand_ns);
    dst.fuzzy_expand_threads = std::max(dst.fuzzy_expand_threads, src.fuzzy_expand_threads);
}

size_t query_manifest_fuzzy_thread_count(size_t fuzzy_row_count)
{
    if (fuzzy_row_count <= 1) {
        return fuzzy_row_count;
    }
    size_t requested = 0;
    if (const char* env = std::getenv("GO_FANN_QUERY_MANIFEST_THREADS")) {
        errno = 0;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(env, &end, 10);
        if (errno == 0 && end != env && *end == '\0' && parsed > 0 &&
            parsed <= std::numeric_limits<size_t>::max()) {
            requested = static_cast<size_t>(parsed);
        }
    }
    if (requested == 0) {
        requested = std::thread::hardware_concurrency();
        if (requested == 0) {
            requested = 1;
        }
        requested = std::min<size_t>(requested, 48);
    }
    return std::max<size_t>(1, std::min(requested, fuzzy_row_count));
}

uint64_t group_layout_checksum(const GroupLayout& group_layout)
{
    uint64_t hash = kFnvOffsetBasis;
    hash = fnv1a_mix_u64(hash, group_layout.groups.size());
    hash = fnv1a_mix_u64(hash, group_layout.label_to_group.size());
    for (const auto& group : group_layout.groups) {
        hash = fnv1a_mix_u64(hash, group.size());
        for (uint32_t label : group) {
            hash = fnv1a_mix_u64(hash, label);
        }
    }
    return hash;
}

SpmatHeaderInfo inspect_spmat_header(const std::string& path, uint32_t row_limit)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open spmat label file " + path);
    }

    SpmatHeaderInfo info;
    input.read(reinterpret_cast<char*>(&info.rows), sizeof(info.rows));
    input.read(reinterpret_cast<char*>(&info.cols), sizeof(info.cols));
    input.read(reinterpret_cast<char*>(&info.nnz), sizeof(info.nnz));
    if (!input) {
        fail_graph("invalid spmat header in " + path);
    }
    if (info.rows > std::numeric_limits<uint32_t>::max()) {
        fail_graph("spmat row count exceeds uint32 range in " + path);
    }
    if (info.cols > std::numeric_limits<uint32_t>::max()) {
        fail_graph("spmat label count exceeds uint32 range in " + path);
    }
    info.effective_rows = (row_limit > 0 && row_limit < info.rows)
        ? row_limit
        : static_cast<uint32_t>(info.rows);
    return info;
}

bool has_matching_highfreq_rows(const GraphMetadata& metadata,
                                const GroupLayout& group_layout,
                                uint32_t original_label)
{
    if (original_label >= group_layout.label_to_group.size()) {
        return false;
    }
    const uint32_t group_id = group_layout.label_to_group[original_label];
    if (group_id == UINT32_MAX || group_id >= metadata.label_size.size()) {
        return false;
    }

    const uint32_t group_size = metadata.label_size[group_id];
    if (group_size == 0) {
        return false;
    }

    const uint64_t filter_bit = metadata.label_filter_bit[original_label];
    if (filter_bit == 0) {
        return false;
    }

    if (metadata.implicit_all_ones_row_filter_mask) {
        return true;
    }

    const uint32_t row_begin = metadata.label_offset[group_id];
    for (uint32_t local_row = 0; local_row < group_size; ++local_row) {
        const uint32_t global_row = row_begin + local_row;
        if ((metadata.row_filter_mask[global_row] & filter_bit) != 0) {
            return true;
        }
    }
    return false;
}

bool graph_trace_enabled()
{
    static const bool enabled = std::getenv("GOFANN_DEBUG_RUNTIME") != nullptr;
    return enabled;
}

void trace_graph_runtime_summary(const GraphRuntime& runtime, const char* stage)
{
    if (!graph_trace_enabled()) {
        return;
    }
    static std::atomic<uint32_t> dump_count{0};
    const uint32_t dump_id = dump_count.fetch_add(1, std::memory_order_relaxed);
    if (dump_id >= 2) {
        return;
    }

    std::fprintf(stdout,
                 "[runtime-trace][runtime] stage=%s label_count=%u total_rows=%u graph_rows=%llu graph_degree=%llu routed_queries=%zu original_queries=%u high=%zu low=%zu query_rows=%u\n",
                 stage,
                 runtime.metadata.label_count,
                 runtime.metadata.total_rows,
                 static_cast<unsigned long long>(runtime.graph.rows),
                 static_cast<unsigned long long>(runtime.graph.cols),
                 runtime.query_group_ids.size(),
                 runtime.original_query_count,
                 runtime.high_query_indices.size(),
                 runtime.low_query_indices.size(),
                 runtime.query_vectors.rows);

    const size_t sample_count = std::min<size_t>(runtime.query_group_ids.size(), 8);
    for (size_t route_idx = 0; route_idx < sample_count; ++route_idx) {
        const uint32_t group_id = runtime.query_group_ids[route_idx];
        const uint32_t label_size = group_id < runtime.metadata.label_size.size()
            ? runtime.metadata.label_size[group_id]
            : UINT32_MAX;
        const uint32_t label_offset = group_id < runtime.metadata.label_offset.size()
            ? runtime.metadata.label_offset[group_id]
            : UINT32_MAX;
        const uint32_t original_query = route_idx < runtime.route_to_original_query.size()
            ? runtime.route_to_original_query[route_idx]
            : UINT32_MAX;
        const uint32_t start_local = route_idx < runtime.start_local_ids.size()
            ? runtime.start_local_ids[route_idx]
            : UINT32_MAX;
        std::fprintf(stdout,
                     "[runtime-trace][runtime] route=%zu orig_query=%u group=%u label_size=%u label_offset=%u start_local=%u lowfreq=%u filter_label=%u\n",
                     route_idx,
                     original_query,
                     group_id,
                     label_size,
                     label_offset,
                     start_local,
                     route_idx < runtime.query_is_lowfreq.size() ? static_cast<unsigned>(runtime.query_is_lowfreq[route_idx]) : 255U,
                     route_idx < runtime.query_filter_labels.size() ? runtime.query_filter_labels[route_idx] : UINT32_MAX);
    }
    std::fflush(stdout);
}

bool is_valid_original_label_seed(const GraphMetadata& metadata,
                                  const GroupLayout& group_layout,
                                  uint32_t original_label,
                                  uint32_t seed)
{
    if (original_label >= group_layout.label_to_group.size()) {
        return false;
    }
    const uint32_t group_id = group_layout.label_to_group[original_label];
    if (group_id == UINT32_MAX || group_id >= metadata.label_size.size()) {
        return false;
    }

    const uint32_t group_size = metadata.label_size[group_id];
    if (group_size == 0) {
        return seed == UINT32_MAX;
    }

    const uint64_t filter_bit = metadata.label_filter_bit[original_label];
    const bool has_rows = has_matching_highfreq_rows(metadata, group_layout, original_label);
    if (!has_rows) {
        return seed == UINT32_MAX;
    }
    if (seed == UINT32_MAX || seed >= group_size) {
        return false;
    }

    if (metadata.implicit_all_ones_row_filter_mask) {
        return true;
    }

    const uint32_t global_row = metadata.label_offset[group_id] + seed;
    return (metadata.row_filter_mask[global_row] & filter_bit) != 0;
}

GroupPostingData build_group_posting_data(const SparseLabelMatrix& data_labels,
                                          const GroupLayout& group_layout)
{
    const auto start = LoadClock::now();
    GroupPostingData out;
    if (data_labels.indptr.size() != static_cast<size_t>(data_labels.rows) + 1) {
        fail_graph("sparse label CSR indptr is not aligned with row count");
    }
    if (group_layout.label_to_group.size() < data_labels.cols) {
        fail_graph("group layout does not cover all sparse label columns");
    }
    out.sample_ids_by_group.resize(group_layout.groups.size());
    out.filter_masks_by_group.resize(group_layout.groups.size());

    std::vector<uint64_t> label_filter_bit(data_labels.cols, 0);
    for (uint32_t group_id = 0; group_id < group_layout.groups.size(); ++group_id) {
        const auto& labels = group_layout.groups[group_id];
        if (labels.size() > 64) {
            fail_graph("group size exceeds uint64 filter mask capacity");
        }
        for (size_t bit_idx = 0; bit_idx < labels.size(); ++bit_idx) {
            const uint32_t original_label = labels[bit_idx];
            if (original_label >= data_labels.cols) {
                fail_graph("group layout label out of range");
            }
            if (group_layout.label_to_group[original_label] != group_id) {
                fail_graph("group layout label-to-group mapping is inconsistent");
            }
            const uint64_t filter_bit = static_cast<uint64_t>(1) << bit_idx;
            if (label_filter_bit[original_label] != 0) {
                fail_graph("group layout repeats label " + std::to_string(original_label));
            }
            label_filter_bit[original_label] = filter_bit;
        }
    }

    std::vector<uint64_t> row_group_masks(group_layout.groups.size(), 0);
    std::vector<uint32_t> touched_groups;
    touched_groups.reserve(32);

    for (uint32_t sample = 0; sample < data_labels.rows; ++sample) {
        const SparseLabelRowSpan row = sparse_label_row_span(data_labels, sample);
        if (!row.valid) {
            fail_graph("sparse label CSR row span is not aligned with row count");
        }
        for (const uint32_t* it = row.begin; it != row.end; ++it) {
            const uint32_t label = *it;
            if (label >= data_labels.cols) {
                fail_graph("sparse label index out of range");
            }
            const uint32_t group_id = group_layout.label_to_group[label];
            if (group_id == UINT32_MAX || group_id >= group_layout.groups.size()) {
                fail_graph("group layout does not map sparse label " + std::to_string(label));
            }
            const uint64_t filter_bit = label_filter_bit[label];
            if (filter_bit == 0) {
                fail_graph("group layout does not assign a filter bit to sparse label " +
                             std::to_string(label));
            }
            uint64_t& mask = row_group_masks[group_id];
            if (mask == 0) {
                touched_groups.push_back(group_id);
            }
            mask |= filter_bit;
        }

        for (uint32_t group_id : touched_groups) {
            out.sample_ids_by_group[group_id].push_back(sample);
            out.filter_masks_by_group[group_id].push_back(row_group_masks[group_id]);
            row_group_masks[group_id] = 0;
        }
        touched_groups.clear();
    }

    log_graph_build_stage("build_group_posting_data", start);
    return out;
}

std::string default_lowfreq_index_path_from_graph_path(const std::string& graph_path)
{
    std::filesystem::path path(graph_path);
    if (path.extension() == ".bin") {
        path.replace_extension(".lowfreq.bin");
        return path.string();
    }
    return graph_path + ".lowfreq.bin";
}

std::string default_graph_file_path_from_graph_path(const std::string& graph_path)
{
    std::filesystem::path path(graph_path);
    if (path.extension() == ".bin") {
        path.replace_extension(".disk.bin");
        return path.string();
    }
    return graph_path + ".disk.bin";
}

struct OrRouteAccuracyCostModel {
    std::string calibration_id;
    double lowfreq_intercept_us = 0.0;
    double lowfreq_per_row_us = 0.0;
    double highfreq_intercept_us = 0.0;
    double highfreq_per_work_us = 0.0;
    double cost_floor_us = 0.0;
    double exact_prior = 0.0;
    double direct_prior = 0.0;
    double fallback_prior = 0.0;
    double fallback_rank_decay = 0.0;
    double base_coverage_weight = 0.0;
    double marginal_coverage_weight = 0.0;
    double marginal_coverage_exponent = 1.0;
    double quality_weight = 0.0;
    double cost_weight = 0.0;
    double latency_scale_us = 0.0;
    double cost_exponent = 1.0;
    uint32_t selection_route_cap = 0;
    uint32_t selection_highfreq_cap = 0;
    uint64_t token_count = 0;
    uint64_t num_points = 0;
    uint64_t max_row_id_exclusive = 0;
    uint64_t sketch_k = 0;
    uint64_t hash_seed = 0;
    uint64_t group_count = 0;
    uint64_t group_layout_checksum = 0;
    std::array<uint8_t, 32> sketch_payload_sha256{};
    std::array<uint8_t, 32> source_posting_sha256{};
};

struct OrRoutePlannerRuntimeSelection {
    OrRouteOrderMode mode = OrRouteOrderMode::Legacy;
    OrRouteOrderMode fuzzy_cap_mode = OrRouteOrderMode::Legacy;
    OrRoutePlannerConfig config;
    OrRouteAccuracyCostModel model;
};

void require_exact_json_fields(const json& object,
                               std::initializer_list<const char*> required,
                               const std::string& context)
{
    if (!object.is_object()) {
        fail_graph(context + " must be a JSON object");
    }
    std::unordered_set<std::string> expected;
    for (const char* field : required) {
        expected.emplace(field);
    }
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (expected.erase(it.key()) == 0) {
            fail_graph(context + " has unknown field '" + it.key() + "'");
        }
    }
    if (!expected.empty()) {
        fail_graph(context + " is missing required field '" + *expected.begin() + "'");
    }
}

double require_finite_json_number(const json& object,
                                  const char* field,
                                  double minimum,
                                  double maximum,
                                  const std::string& context)
{
    if (!object.at(field).is_number()) {
        fail_graph(context + "." + field + " must be numeric");
    }
    const double value = object.at(field).get<double>();
    if (!std::isfinite(value) || value < minimum || value > maximum) {
        fail_graph(context + "." + field + " is outside its finite valid range");
    }
    return value;
}

uint64_t require_json_u64(const json& object,
                          const char* field,
                          uint64_t minimum,
                          const std::string& context)
{
    if (!object.at(field).is_number_unsigned()) {
        fail_graph(context + "." + field + " must be an unsigned integer");
    }
    const uint64_t value = object.at(field).get<uint64_t>();
    if (value < minimum) {
        fail_graph(context + "." + field + " is below its minimum");
    }
    return value;
}

void require_json_ratio_matches(double numerator,
                                double denominator,
                                double ratio,
                                const std::string& context)
{
    if (denominator <= 0.0) {
        fail_graph(context + " denominator must be positive");
    }
    const double expected = numerator / denominator;
    const double tolerance = 1.0e-9 * std::max({1.0, std::abs(expected), std::abs(ratio)});
    if (!std::isfinite(expected) || std::abs(expected - ratio) > tolerance) {
        fail_graph(context + " does not match its numerator/baseline values");
    }
}

std::array<uint8_t, 32> parse_sha256_hex(const json& value, const std::string& context)
{
    if (!value.is_string()) {
        fail_graph(context + " must be a 64-character hexadecimal string");
    }
    const std::string hex = value.get<std::string>();
    if (hex.size() != 64) {
        fail_graph(context + " must be a 64-character hexadecimal string");
    }
    std::array<uint8_t, 32> bytes{};
    for (size_t idx = 0; idx < bytes.size(); ++idx) {
        const auto nibble = [&](char ch) -> uint8_t {
            if (ch >= '0' && ch <= '9') {
                return static_cast<uint8_t>(ch - '0');
            }
            if (ch >= 'a' && ch <= 'f') {
                return static_cast<uint8_t>(ch - 'a' + 10);
            }
            if (ch >= 'A' && ch <= 'F') {
                return static_cast<uint8_t>(ch - 'A' + 10);
            }
            fail_graph(context + " contains a non-hexadecimal character");
        };
        bytes[idx] = static_cast<uint8_t>((nibble(hex[idx * 2]) << 4) |
                                          nibble(hex[idx * 2 + 1]));
    }
    return bytes;
}

OrRouteAccuracyCostModel load_or_route_accuracy_cost_model(const std::string& path)
{
    std::ifstream input(path);
    if (!input) {
        fail_graph("unable to open OR route accuracy/cost model " + path);
    }
    json root;
    try {
        input >> root;
    } catch (const json::exception& error) {
        fail_graph("invalid OR route accuracy/cost model JSON in " + path + ": " + error.what());
    }
    require_exact_json_fields(
        root,
        {"schema", "version", "calibration_id", "validated",
         "training_sample_count", "training_query_count", "validation_sample_count",
         "validation_query_count", "calibration_run_count", "cost_model",
         "quality_model", "utility", "fit_diagnostics", "sketch", "provenance"},
        "OR route accuracy/cost model");
    if (!root.at("schema").is_string() ||
        root.at("schema").get<std::string>() != "flashanns.or_route_accuracy_cost_model") {
        fail_graph("unsupported OR route accuracy/cost model schema");
    }
    if (!root.at("version").is_number_unsigned() || root.at("version").get<uint32_t>() != 1) {
        fail_graph("unsupported OR route accuracy/cost model version");
    }
    if (!root.at("validated").is_boolean() || !root.at("validated").get<bool>()) {
        fail_graph("OR route accuracy/cost model must have validated=true");
    }
    if (!root.at("calibration_id").is_string() ||
        root.at("calibration_id").get<std::string>().empty()) {
        fail_graph("OR route accuracy/cost model calibration_id must be nonempty");
    }
    require_json_u64(root, "training_sample_count", 1000, "OR route accuracy/cost model");
    require_json_u64(root, "training_query_count", 100, "OR route accuracy/cost model");
    require_json_u64(root, "validation_sample_count", 250, "OR route accuracy/cost model");
    require_json_u64(root, "validation_query_count", 25, "OR route accuracy/cost model");
    require_json_u64(root, "calibration_run_count", 3, "OR route accuracy/cost model");

    const json& cost = root.at("cost_model");
    require_exact_json_fields(cost,
                              {"latency_unit", "lowfreq_intercept_us", "lowfreq_per_row_us",
                               "highfreq_intercept_us", "highfreq_per_work_us", "cost_floor_us"},
                              "cost_model");
    if (!cost.at("latency_unit").is_string() ||
        cost.at("latency_unit").get<std::string>() != "microseconds") {
        fail_graph("cost_model.latency_unit must be 'microseconds'");
    }

    const json& quality = root.at("quality_model");
    require_exact_json_fields(quality,
                              {"exact_prior", "direct_prior", "fallback_prior",
                               "fallback_rank_decay", "base_coverage_weight",
                               "marginal_coverage_weight",
                               "marginal_coverage_exponent"},
                              "quality_model");
    const json& utility = root.at("utility");
    require_exact_json_fields(utility,
                              {"formula", "quality_weight", "cost_weight", "latency_scale_us",
                               "cost_exponent"},
                              "utility");
    constexpr const char* kExpectedFormula =
        "quality_weight*quality-cost_weight*(max(predicted_latency_us,cost_floor_us)/latency_scale_us)^cost_exponent";
    if (!utility.at("formula").is_string() ||
        utility.at("formula").get<std::string>() != kExpectedFormula) {
        fail_graph("utility.formula is unsupported or ambiguous");
    }

    const json& fit = root.at("fit_diagnostics");
    require_exact_json_fields(fit,
                              {"heldout_query_fraction", "cost_r2", "quality_r2", "cost_mae_us",
                               "quality_mae", "split_seed", "lowfreq_condition_number",
                               "highfreq_condition_number", "quality_condition_number",
                               "selection_route_cap", "selection_highfreq_cap",
                               "utility_search_candidates", "heldout_gt_coverage",
                               "heldout_baseline_gt_coverage", "heldout_gt_coverage_ratio",
                               "heldout_predicted_latency_us",
                               "heldout_baseline_predicted_latency_us",
                               "heldout_predicted_latency_ratio",
                               "min_required_gt_coverage_ratio", "max_allowed_latency_ratio",
                               "validation_errors"},
                              "fit_diagnostics");
    require_finite_json_number(fit, "heldout_query_fraction", 0.01, 0.5, "fit_diagnostics");
    require_finite_json_number(fit, "cost_r2", 0.5, 1.0, "fit_diagnostics");
    require_finite_json_number(fit, "quality_r2", 0.05, 1.0, "fit_diagnostics");
    require_finite_json_number(fit, "cost_mae_us", 0.0, 1.0e12, "fit_diagnostics");
    require_finite_json_number(fit, "quality_mae", 0.0, 1.0e12, "fit_diagnostics");
    require_json_u64(fit, "split_seed", 0, "fit_diagnostics");
    require_finite_json_number(
        fit, "lowfreq_condition_number", 1.0, 1.0e8, "fit_diagnostics");
    require_finite_json_number(
        fit, "highfreq_condition_number", 1.0, 1.0e8, "fit_diagnostics");
    require_finite_json_number(
        fit, "quality_condition_number", 1.0, 1.0e8, "fit_diagnostics");
    const uint64_t selection_route_cap = require_json_u64(
        fit, "selection_route_cap", 0, "fit_diagnostics");
    const uint64_t selection_highfreq_cap = require_json_u64(
        fit, "selection_highfreq_cap", 0, "fit_diagnostics");
    const uint64_t utility_search_candidates = require_json_u64(
        fit, "utility_search_candidates", 2, "fit_diagnostics");
    if (selection_route_cap > std::numeric_limits<uint32_t>::max() ||
        selection_highfreq_cap > std::numeric_limits<uint32_t>::max() ||
        utility_search_candidates > std::numeric_limits<uint32_t>::max()) {
        fail_graph("fit_diagnostics selection/search counts exceed uint32 range");
    }
    if (selection_route_cap > 0 && selection_highfreq_cap > selection_route_cap) {
        fail_graph(
            "fit_diagnostics.selection_highfreq_cap exceeds selection_route_cap");
    }
    const double heldout_gt_coverage = static_cast<double>(require_json_u64(
        fit, "heldout_gt_coverage", 0, "fit_diagnostics"));
    const double heldout_baseline_gt_coverage = static_cast<double>(require_json_u64(
        fit, "heldout_baseline_gt_coverage", 1, "fit_diagnostics"));
    const double heldout_gt_coverage_ratio = require_finite_json_number(
        fit, "heldout_gt_coverage_ratio", 0.0, 10.0, "fit_diagnostics");
    const double heldout_predicted_latency_us = require_finite_json_number(
        fit,
        "heldout_predicted_latency_us",
        std::numeric_limits<double>::min(),
        1.0e12,
        "fit_diagnostics");
    const double heldout_baseline_predicted_latency_us = require_finite_json_number(
        fit,
        "heldout_baseline_predicted_latency_us",
        std::numeric_limits<double>::min(),
        1.0e12,
        "fit_diagnostics");
    const double heldout_predicted_latency_ratio = require_finite_json_number(
        fit,
        "heldout_predicted_latency_ratio",
        std::numeric_limits<double>::min(),
        10.0,
        "fit_diagnostics");
    const double min_required_gt_coverage_ratio = require_finite_json_number(
        fit, "min_required_gt_coverage_ratio", 0.0, 1.0, "fit_diagnostics");
    const double max_allowed_latency_ratio = require_finite_json_number(
        fit,
        "max_allowed_latency_ratio",
        std::numeric_limits<double>::min(),
        10.0,
        "fit_diagnostics");
    require_json_ratio_matches(heldout_gt_coverage,
                               heldout_baseline_gt_coverage,
                               heldout_gt_coverage_ratio,
                               "fit_diagnostics.heldout_gt_coverage_ratio");
    require_json_ratio_matches(heldout_predicted_latency_us,
                               heldout_baseline_predicted_latency_us,
                               heldout_predicted_latency_ratio,
                               "fit_diagnostics.heldout_predicted_latency_ratio");
    if (heldout_gt_coverage_ratio < min_required_gt_coverage_ratio) {
        fail_graph(
            "fit_diagnostics.heldout_gt_coverage_ratio is below the required minimum");
    }
    if (heldout_predicted_latency_ratio > max_allowed_latency_ratio) {
        fail_graph(
            "fit_diagnostics.heldout_predicted_latency_ratio exceeds the allowed maximum");
    }
    if (!fit.at("validation_errors").is_array() ||
        !fit.at("validation_errors").empty()) {
        fail_graph("validated model fit_diagnostics.validation_errors must be empty");
    }

    const json& sketch = root.at("sketch");
    require_exact_json_fields(sketch,
                              {"sidecar_schema", "sidecar_version", "token_count", "num_points",
                               "max_row_id_exclusive", "sketch_k", "hash_seed", "group_count",
                               "group_layout_checksum", "payload_sha256",
                               "source_posting_sha256"},
                              "sketch");
    if (!sketch.at("sidecar_schema").is_string() ||
        sketch.at("sidecar_schema").get<std::string>() != "flashanns.or_route_kmv_sidecar") {
        fail_graph("unsupported sketch.sidecar_schema");
    }
    if (!sketch.at("sidecar_version").is_number_unsigned() ||
        sketch.at("sidecar_version").get<uint32_t>() != kOrRouteSketchSidecarVersion) {
        fail_graph("unsupported sketch.sidecar_version");
    }

    const json& provenance = root.at("provenance");
    require_exact_json_fields(provenance,
                              {"tool", "created_utc", "trace_inputs", "source"},
                              "provenance");
    if (!provenance.at("tool").is_string() ||
        !provenance.at("created_utc").is_string() ||
        !provenance.at("source").is_string() ||
        !provenance.at("trace_inputs").is_array()) {
        fail_graph("provenance fields have invalid types");
    }
    for (const json& trace_input : provenance.at("trace_inputs")) {
        if (!trace_input.is_string()) {
            fail_graph("provenance.trace_inputs must contain only strings");
        }
    }

    OrRouteAccuracyCostModel model;
    model.calibration_id = root.at("calibration_id").get<std::string>();
    model.lowfreq_intercept_us = require_finite_json_number(
        cost, "lowfreq_intercept_us", 0.0, 1.0e12, "cost_model");
    model.lowfreq_per_row_us = require_finite_json_number(
        cost, "lowfreq_per_row_us", std::numeric_limits<double>::min(), 1.0e12, "cost_model");
    model.highfreq_intercept_us = require_finite_json_number(
        cost, "highfreq_intercept_us", 0.0, 1.0e12, "cost_model");
    model.highfreq_per_work_us = require_finite_json_number(
        cost, "highfreq_per_work_us", std::numeric_limits<double>::min(), 1.0e12, "cost_model");
    model.cost_floor_us = require_finite_json_number(
        cost, "cost_floor_us", std::numeric_limits<double>::min(), 1.0e12, "cost_model");
    model.exact_prior = require_finite_json_number(
        quality, "exact_prior", std::numeric_limits<double>::min(), 100.0, "quality_model");
    model.direct_prior = require_finite_json_number(
        quality, "direct_prior", std::numeric_limits<double>::min(), 100.0, "quality_model");
    model.fallback_prior = require_finite_json_number(
        quality, "fallback_prior", std::numeric_limits<double>::min(), 100.0, "quality_model");
    model.fallback_rank_decay = require_finite_json_number(
        quality, "fallback_rank_decay", std::numeric_limits<double>::min(), 10.0, "quality_model");
    model.base_coverage_weight = require_finite_json_number(
        quality, "base_coverage_weight", std::numeric_limits<double>::min(), 100.0, "quality_model");
    model.marginal_coverage_weight = require_finite_json_number(
        quality, "marginal_coverage_weight", 0.0, 100.0, "quality_model");
    model.marginal_coverage_exponent = require_finite_json_number(
        quality, "marginal_coverage_exponent", 0.1, 4.0, "quality_model");
    model.quality_weight = require_finite_json_number(
        utility, "quality_weight", std::numeric_limits<double>::min(), 100.0, "utility");
    model.cost_weight = require_finite_json_number(
        utility, "cost_weight", std::numeric_limits<double>::min(), 100.0, "utility");
    model.latency_scale_us = require_finite_json_number(
        utility, "latency_scale_us", std::numeric_limits<double>::min(), 1.0e12, "utility");
    model.cost_exponent = require_finite_json_number(
        utility, "cost_exponent", 0.1, 4.0, "utility");
    model.selection_route_cap = static_cast<uint32_t>(selection_route_cap);
    model.selection_highfreq_cap = static_cast<uint32_t>(selection_highfreq_cap);
    model.token_count = require_json_u64(sketch, "token_count", 1, "sketch");
    model.num_points = require_json_u64(sketch, "num_points", 1, "sketch");
    model.max_row_id_exclusive = require_json_u64(
        sketch, "max_row_id_exclusive", 1, "sketch");
    model.sketch_k = require_json_u64(sketch, "sketch_k", kOrRouteSketchMinK, "sketch");
    if (model.sketch_k > kOrRouteSketchMaxK) {
        fail_graph("sketch.sketch_k exceeds the supported maximum");
    }
    model.hash_seed = require_json_u64(sketch, "hash_seed", 0, "sketch");
    model.group_count = require_json_u64(sketch, "group_count", 1, "sketch");
    model.group_layout_checksum = require_json_u64(
        sketch, "group_layout_checksum", 0, "sketch");
    model.sketch_payload_sha256 = parse_sha256_hex(
        sketch.at("payload_sha256"), "sketch.payload_sha256");
    model.source_posting_sha256 = parse_sha256_hex(
        sketch.at("source_posting_sha256"), "sketch.source_posting_sha256");
    if (model.num_points != model.max_row_id_exclusive) {
        fail_graph("sketch num_points and max_row_id_exclusive must match for v1");
    }
    return model;
}

std::mutex& or_route_planner_selection_mutex()
{
    static std::mutex mutex;
    return mutex;
}

OrRoutePlannerRuntimeSelection& or_route_planner_selection()
{
    static OrRoutePlannerRuntimeSelection selection;
    return selection;
}

std::optional<OrRouteOrderMode> parse_or_route_order_mode(std::string mode)
{
    if (mode.empty()) {
        return OrRouteOrderMode::Legacy;
    }
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch) {
        if (ch == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(ch));
    });
    if (mode == "legacy" || mode == "manifest") {
        return OrRouteOrderMode::Legacy;
    }
    if (mode == "cost_aware") {
        return OrRouteOrderMode::CostAware;
    }
    if (mode == "coverage_cost") {
        return OrRouteOrderMode::CoverageCost;
    }
    if (mode == "relevance_coverage_cost") {
        return OrRouteOrderMode::RelevanceCoverageCost;
    }
    if (mode == "marginal_accuracy_cost") {
        return OrRouteOrderMode::MarginalAccuracyCost;
    }
    return std::nullopt;
}

std::optional<QueryManifestOperatorFillMode> parse_query_manifest_operator_fill_mode(
    std::string mode)
{
    if (mode.empty()) {
        return QueryManifestOperatorFillMode::LegacyFullFallback;
    }
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char ch) {
        if (ch == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(ch));
    });
    if (mode == "legacy_full_fallback") {
        return QueryManifestOperatorFillMode::LegacyFullFallback;
    }
    if (mode == "direct_then_fallback") {
        return QueryManifestOperatorFillMode::DirectThenFallback;
    }
    if (mode == "literal_only") {
        return QueryManifestOperatorFillMode::LiteralOnly;
    }
    return std::nullopt;
}

OrRoutePlannerRuntimeSelection snapshot_or_route_planner_selection()
{
    std::lock_guard<std::mutex> lock(or_route_planner_selection_mutex());
    return or_route_planner_selection();
}

}

const char* query_label_source_tier_name(QueryLabelSourceTier tier)
{
    switch (tier) {
    case QueryLabelSourceTier::Direct:
        return "direct";
    case QueryLabelSourceTier::Fallback:
        return "fallback";
    case QueryLabelSourceTier::Exact:
    default:
        return "exact";
    }
}

const char* query_manifest_operator_fill_mode_name(QueryManifestOperatorFillMode mode)
{
    switch (mode) {
    case QueryManifestOperatorFillMode::LiteralOnly:
        return "literal_only";
    case QueryManifestOperatorFillMode::DirectThenFallback:
        return "direct_then_fallback";
    case QueryManifestOperatorFillMode::LegacyFullFallback:
    default:
        return "legacy_full_fallback";
    }
}

bool is_valid_query_manifest_operator_fill_mode(const std::string& mode)
{
    return parse_query_manifest_operator_fill_mode(mode).has_value();
}

bool query_manifest_operator_fill_mode_uses_nested_fill(const std::string& mode)
{
    const std::optional<QueryManifestOperatorFillMode> parsed =
        parse_query_manifest_operator_fill_mode(mode);
    return parsed.has_value() &&
           *parsed == QueryManifestOperatorFillMode::DirectThenFallback;
}

bool query_manifest_operator_fill_mode_requires_operator_index(const std::string& mode)
{
    const std::optional<QueryManifestOperatorFillMode> parsed =
        parse_query_manifest_operator_fill_mode(mode);
    return parsed.has_value() &&
           *parsed != QueryManifestOperatorFillMode::LegacyFullFallback;
}

void validate_query_manifest_operator_rewrite_options(const std::string& fill_mode,
                                                      bool parity_check)
{
    const std::optional<QueryManifestOperatorFillMode> parsed =
        parse_query_manifest_operator_fill_mode(fill_mode);
    if (!parsed.has_value()) {
        fail_graph(
            "query_manifest_operator_fill_mode must be one of: "
            "legacy_full_fallback, direct_then_fallback, literal_only");
    }
    if (*parsed != QueryManifestOperatorFillMode::LegacyFullFallback && parity_check) {
        fail_graph(
            "query_manifest_operator_fill_mode=" +
            std::string(query_manifest_operator_fill_mode_name(*parsed)) + " conflicts with "
            "query_manifest_operator_parity_check=true");
    }
}

void validate_query_manifest_fuzzy_operator_fill_mode(const std::string& fill_mode)
{
    const std::optional<QueryManifestOperatorFillMode> parsed =
        parse_query_manifest_operator_fill_mode(fill_mode);
    if (!parsed.has_value()) {
        fail_graph("invalid query manifest fuzzy operator fill mode " + fill_mode);
    }
    if (*parsed != QueryManifestOperatorFillMode::LiteralOnly) {
        fail_graph(
            "manifest fuzzy semantics require "
            "query_manifest_operator_fill_mode=literal_only; fallback tokens do not "
            "satisfy prefix, suffix, or contains predicates");
    }
}

const char* or_route_order_mode_name(OrRouteOrderMode mode)
{
    switch (mode) {
    case OrRouteOrderMode::MarginalAccuracyCost:
        return "marginal_accuracy_cost";
    case OrRouteOrderMode::RelevanceCoverageCost:
        return "relevance_coverage_cost";
    case OrRouteOrderMode::CoverageCost:
        return "coverage_cost";
    case OrRouteOrderMode::CostAware:
        return "cost_aware";
    case OrRouteOrderMode::Legacy:
    default:
        return "legacy";
    }
}

bool is_valid_or_route_order_mode(const std::string& mode)
{
    return parse_or_route_order_mode(mode).has_value();
}

bool is_valid_query_manifest_fuzzy_cap_policy(const std::string& policy)
{
    std::string normalized = policy;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        if (ch == '-') {
            return '_';
        }
        return static_cast<char>(std::tolower(ch));
    });
    return normalized == "legacy" || normalized == "cost_aware" ||
           normalized == "coverage_cost" ||
           normalized == "relevance_coverage_cost";
}

void configure_query_preparation(const QueryPreparationConfig& config)
{
    if (config.threads > 256) fail_graph("query_prepare_threads must be in [0, 256]");
    std::lock_guard<std::mutex> lock(query_preparation_config_mutex);
    query_preparation_config = config;
}

void configure_or_route_planner(const std::string& order_mode)
{
    OrRoutePlannerConfig config;
    config.order_mode = order_mode;
    configure_or_route_planner(config);
}

void configure_or_route_planner(const OrRoutePlannerConfig& config)
{
    const std::optional<OrRouteOrderMode> parsed = parse_or_route_order_mode(config.order_mode);
    if (!parsed.has_value()) {
        fail_graph(
            "or_route_order_mode must be one of: legacy, manifest, cost_aware, "
            "coverage_cost, relevance_coverage_cost, marginal_accuracy_cost");
    }
    if (!is_valid_query_manifest_fuzzy_cap_policy(config.fuzzy_cap_policy)) {
        fail_graph(
            "query_manifest_fuzzy_cap_policy must be one of: legacy, cost_aware, "
            "coverage_cost, relevance_coverage_cost");
    }
    const OrRouteOrderMode fuzzy_cap_mode =
        *parse_or_route_order_mode(config.fuzzy_cap_policy);
    if (config.sketch_k < kOrRouteSketchMinK || config.sketch_k > kOrRouteSketchMaxK) {
        fail_graph("or_route_sketch_k must be in [4, 4096]");
    }
    if (config.runtime_sketch_fallback_max_rows == 0 ||
        config.runtime_sketch_fallback_max_rows > 1000000) {
        fail_graph("runtime sketch fallback max rows must be in [1, 1000000]");
    }
    if (!config.calibration_trace_path.empty() && config.calibration_run_id.empty()) {
        fail_graph(
            "or_route_calibration_trace_path requires a nonempty calibration run id");
    }
    OrRoutePlannerRuntimeSelection selection;
    selection.mode = *parsed;
    selection.fuzzy_cap_mode = fuzzy_cap_mode;
    selection.config = config;
    if (*parsed == OrRouteOrderMode::MarginalAccuracyCost) {
        if (config.accuracy_cost_model_path.empty()) {
            fail_graph("marginal_accuracy_cost requires an explicit accuracy/cost model path");
        }
        if (config.sketch_sidecar_path.empty() && !config.allow_runtime_sketch_fallback) {
            fail_graph(
                "marginal_accuracy_cost requires a sketch sidecar; runtime fallback is only for explicit small fixtures");
        }
        selection.model = load_or_route_accuracy_cost_model(config.accuracy_cost_model_path);
        if (selection.model.sketch_k != config.sketch_k ||
            selection.model.hash_seed != config.sketch_hash_seed) {
            fail_graph("planner sketch K/hash seed do not match the calibrated model");
        }
    }
    std::lock_guard<std::mutex> lock(or_route_planner_selection_mutex());
    or_route_planner_selection() = std::move(selection);
}

size_t QueryRangeTable::flat_index(uint32_t row, uint32_t range_idx) const
{
    return static_cast<size_t>(row) * range_count + range_idx;
}

const char* dense_vector_scalar_type_name(DenseVectorScalarType type)
{
    switch (type) {
    case DenseVectorScalarType::Uint8:
        return "uint8";
    case DenseVectorScalarType::Float32:
        return "float32";
    case DenseVectorScalarType::Float16:
        return "float16";
    case DenseVectorScalarType::Unknown:
    default:
        return "unknown";
    }
}

size_t dense_vector_scalar_size(DenseVectorScalarType type)
{
    switch (type) {
    case DenseVectorScalarType::Uint8:
        return sizeof(uint8_t);
    case DenseVectorScalarType::Float32:
        return sizeof(float);
    case DenseVectorScalarType::Float16:
        return sizeof(uint16_t);
    case DenseVectorScalarType::Unknown:
    default:
        fail_graph("unknown dense vector scalar type");
    }
}

bool dense_vector_scalar_is_float(DenseVectorScalarType type)
{
    return type == DenseVectorScalarType::Float32 ||
           type == DenseVectorScalarType::Float16;
}

const uint8_t* DenseUint8Matrix::row_ptr(uint32_t row) const
{
    return data.data() + static_cast<size_t>(row) * cols;
}

const float* DenseFloatMatrix::row_ptr(uint32_t row) const
{
    return data.data() + static_cast<size_t>(row) * cols;
}

std::string lower_extension(const std::string& path)
{
    std::string ext = std::filesystem::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

namespace {

constexpr uint64_t kFbinHeaderBytes = 2 * sizeof(uint32_t);

struct FbinHeader {
    uint32_t rows = 0;
    uint32_t cols = 0;
};

FbinHeader inspect_fbin_header(std::ifstream& input, const std::string& path)
{
    FbinHeader header;
    input.read(reinterpret_cast<char*>(&header.rows), sizeof(header.rows));
    input.read(reinterpret_cast<char*>(&header.cols), sizeof(header.cols));
    if (!input || header.rows == 0 || header.cols == 0) {
        fail_graph("invalid fbin header in " + path);
    }

    const uint64_t value_count = static_cast<uint64_t>(header.rows) * header.cols;
    if (value_count > (std::numeric_limits<uint64_t>::max() - kFbinHeaderBytes) /
                          sizeof(float)) {
        fail_graph("fbin file size exceeds uint64 range for " + path);
    }
    const uint64_t expected_size = kFbinHeaderBytes + value_count * sizeof(float);
    if (expected_size > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        fail_graph("fbin file size exceeds stream offset range for " + path);
    }
    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    if (file_size < 0 || static_cast<uint64_t>(file_size) != expected_size) {
        fail_graph("fbin file size does not match header in " + path);
    }
    return header;
}

}  // namespace

DenseVectorMetadata inspect_dense_vector_file(const std::string& path, uint32_t row_limit)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open vector file " + path);
    }

    DenseVectorMetadata metadata;
    metadata.path = path;
    const std::string ext = lower_extension(path);
    if (ext == ".ibin" || ext == ".u8bin") {
        uint32_t rows = 0;
        uint32_t cols = 0;
        input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        if (!input || cols == 0) {
            fail_graph("invalid uint8 vector header in " + path);
        }
        metadata.scalar_type = DenseVectorScalarType::Uint8;
        metadata.rows = (row_limit > 0 && row_limit < rows) ? row_limit : rows;
        metadata.cols = cols;
        return metadata;
    }

    if (ext == ".fbin") {
        const FbinHeader header = inspect_fbin_header(input, path);
        metadata.scalar_type = DenseVectorScalarType::Float32;
        metadata.rows = (row_limit > 0 && row_limit < header.rows) ? row_limit : header.rows;
        metadata.cols = header.cols;
        return metadata;
    }

    if (ext == ".bvecs" || ext == ".fvecs" || ext == ".hvecs") {
        uint32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!input || dim == 0) {
            fail_graph("invalid " + ext + " dimension in " + path);
        }
        input.seekg(0, std::ios::end);
        const std::streamoff file_size = input.tellg();
        std::streamoff payload_bytes = static_cast<std::streamoff>(dim);
        if (ext == ".fvecs") {
            payload_bytes *= static_cast<std::streamoff>(sizeof(float));
        } else if (ext == ".hvecs") {
            payload_bytes *= static_cast<std::streamoff>(sizeof(uint16_t));
        }
        const std::streamoff record_size = static_cast<std::streamoff>(sizeof(uint32_t)) + payload_bytes;
        if (record_size <= 0 || file_size < 0 || file_size % record_size != 0) {
            fail_graph(ext + " file size is not row-aligned for " + path);
        }
        const uint64_t total_rows64 = static_cast<uint64_t>(file_size / record_size);
        if (total_rows64 > std::numeric_limits<uint32_t>::max()) {
            fail_graph(ext + " row count exceeds uint32 range for " + path);
        }
        const uint32_t total_rows = static_cast<uint32_t>(total_rows64);
        if (ext == ".bvecs") {
            metadata.scalar_type = DenseVectorScalarType::Uint8;
        } else if (ext == ".hvecs") {
            metadata.scalar_type = DenseVectorScalarType::Float16;
        } else {
            metadata.scalar_type = DenseVectorScalarType::Float32;
        }
        metadata.rows = (row_limit > 0 && row_limit < total_rows) ? row_limit : total_rows;
        metadata.cols = dim;
        return metadata;
    }

    fail_graph("unsupported dense vector extension for " + path +
                 " (supported: .bvecs, .ibin, .u8bin, .fbin, .fvecs, .hvecs)");
}

DenseUint8Matrix load_uint8_ibin_like(const std::string& path, uint32_t row_limit)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open vector file " + path);
    }

    uint32_t rows = 0;
    uint32_t cols = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!input) {
        fail_graph("failed to read header from " + path);
    }

    const uint32_t effective_rows = (row_limit > 0 && row_limit < rows) ? row_limit : rows;
    DenseUint8Matrix matrix;
    matrix.rows = effective_rows;
    matrix.cols = cols;
    matrix.data.resize(static_cast<size_t>(effective_rows) * cols);

    input.read(reinterpret_cast<char*>(matrix.data.data()), matrix.data.size() * sizeof(uint8_t));
    if (!input) {
        fail_graph("failed to read payload from " + path);
    }
    return matrix;
}

DenseUint8Matrix load_bvecs_matrix(const std::string& path, uint32_t row_limit)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open bvecs file " + path);
    }

    uint32_t dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!input || dim == 0) {
        fail_graph("invalid bvecs dimension in " + path);
    }

    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    const std::streamoff record_size = sizeof(uint32_t) + static_cast<std::streamoff>(dim);
    if (record_size <= 0 || file_size % record_size != 0) {
        fail_graph("bvecs file size is not row-aligned for " + path);
    }

    const uint32_t total_rows = static_cast<uint32_t>(file_size / record_size);
    const uint32_t effective_rows = (row_limit > 0 && row_limit < total_rows) ? row_limit : total_rows;

    DenseUint8Matrix matrix;
    matrix.rows = effective_rows;
    matrix.cols = dim;
    matrix.data.resize(static_cast<size_t>(effective_rows) * dim);

    input.seekg(0, std::ios::beg);
    for (uint32_t row = 0; row < effective_rows; ++row) {
        uint32_t row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        if (!input || row_dim != dim) {
            fail_graph("inconsistent bvecs dimension at row " + std::to_string(row) + " in " + path);
        }
        input.read(reinterpret_cast<char*>(matrix.data.data() + static_cast<size_t>(row) * dim), dim);
        if (!input) {
            fail_graph("failed to read bvecs payload from " + path);
        }
    }

    return matrix;
}

DenseUint8Matrix load_uint8_matrix_auto(const std::string& path, uint32_t row_limit)
{
    const std::string ext = lower_extension(path);
    if (ext == ".bvecs") {
        return load_bvecs_matrix(path, row_limit);
    }
    if (ext == ".ibin" || ext == ".u8bin") {
        return load_uint8_ibin_like(path, row_limit);
    }
    if (ext == ".fbin" || ext == ".fvecs") {
        fail_graph("cannot load float32 " + ext + " file as uint8 vectors: " + path);
    }
    if (ext == ".hvecs") {
        fail_graph("cannot load float16 .hvecs file as uint8 vectors: " + path);
    }
    fail_graph("unsupported uint8 vector file extension for " + path +
                 " (supported: .bvecs, .ibin, .u8bin)");
}

DenseFloatMatrix load_fvecs_matrix(const std::string& path, uint32_t row_limit)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open float vector file " + path);
    }

    const std::string ext = lower_extension(path);
    if (ext != ".fvecs" && ext != ".hvecs") {
        fail_graph("unsupported float vector file extension for " + path +
                     " (supported: .fvecs, .hvecs)");
    }

    uint32_t dim = 0;
    input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!input || dim == 0) {
        fail_graph("invalid " + ext + " dimension in " + path);
    }

    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    const std::streamoff record_size =
        static_cast<std::streamoff>(sizeof(uint32_t)) +
        static_cast<std::streamoff>(dim) *
            static_cast<std::streamoff>(ext == ".hvecs" ? sizeof(uint16_t) : sizeof(float));
    if (record_size <= 0 || file_size < 0 || file_size % record_size != 0) {
        fail_graph(ext + " file size is not row-aligned for " + path);
    }

    const uint64_t total_rows64 = static_cast<uint64_t>(file_size / record_size);
    if (total_rows64 > std::numeric_limits<uint32_t>::max()) {
        fail_graph(ext + " row count exceeds uint32 range for " + path);
    }
    const uint32_t total_rows = static_cast<uint32_t>(total_rows64);
    const uint32_t effective_rows = (row_limit > 0 && row_limit < total_rows) ? row_limit : total_rows;

    DenseFloatMatrix matrix;
    matrix.rows = effective_rows;
    matrix.cols = dim;
    matrix.data.resize(static_cast<size_t>(effective_rows) * dim);

    input.seekg(0, std::ios::beg);
    for (uint32_t row = 0; row < effective_rows; ++row) {
        uint32_t row_dim = 0;
        input.read(reinterpret_cast<char*>(&row_dim), sizeof(row_dim));
        if (!input || row_dim != dim) {
            fail_graph("inconsistent " + ext + " dimension at row " + std::to_string(row) + " in " + path);
        }
        float* dst = matrix.data.data() + static_cast<size_t>(row) * dim;
        if (ext == ".hvecs") {
            std::vector<uint16_t> half_row(dim);
            input.read(reinterpret_cast<char*>(half_row.data()),
                       static_cast<std::streamsize>(sizeof(uint16_t) * static_cast<size_t>(dim)));
            if (!input) {
                fail_graph("failed to read hvecs payload from " + path);
            }
            convert_half_row_to_float(half_row.data(), dst, dim);
        } else {
            input.read(reinterpret_cast<char*>(dst),
                       static_cast<std::streamsize>(sizeof(float) * static_cast<size_t>(dim)));
            if (!input) {
                fail_graph("failed to read fvecs payload from " + path);
            }
        }
    }

    return matrix;
}

DenseFloatMatrix load_float_matrix_auto(const std::string& path, uint32_t row_limit)
{
    note_full_base_loader_call("load_float_matrix_auto", path);
    const std::string ext = lower_extension(path);
    if (ext == ".fbin") {
        std::ifstream input(path, std::ios::binary);
        if (!input) {
            fail_graph("unable to open float vector file " + path);
        }
        const FbinHeader header = inspect_fbin_header(input, path);
        DenseFloatMatrix matrix;
        matrix.rows = (row_limit > 0 && row_limit < header.rows) ? row_limit : header.rows;
        matrix.cols = header.cols;
        const uint64_t value_count = static_cast<uint64_t>(matrix.rows) * matrix.cols;
        if (value_count > std::numeric_limits<size_t>::max() / sizeof(float) ||
            value_count > matrix.data.max_size() ||
            value_count > static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()) /
                              sizeof(float)) {
            fail_graph("fbin payload exceeds host allocation or stream size range for " + path);
        }
        matrix.data.resize(static_cast<size_t>(value_count));
        input.seekg(kFbinHeaderBytes, std::ios::beg);
        input.read(reinterpret_cast<char*>(matrix.data.data()),
                   static_cast<std::streamsize>(value_count * sizeof(float)));
        if (!input) {
            fail_graph("failed to read fbin payload from " + path);
        }
        return matrix;
    }
    if (ext == ".fvecs" || ext == ".hvecs") {
        return load_fvecs_matrix(path, row_limit);
    }
    fail_graph("unsupported float vector file extension for " + path +
                 " (supported: .fbin, .fvecs, .hvecs)");
}

VectorFileInfo inspect_vector_file(const std::string& path, uint32_t row_limit)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open vector file " + path);
    }

    VectorFileInfo info;
    info.path = path;
    const std::string ext = lower_extension(path);
    if (ext == ".fbin") {
        const FbinHeader header = inspect_fbin_header(input, path);
        const uint64_t row_bytes = static_cast<uint64_t>(header.cols) * sizeof(float);
        if (row_bytes > std::numeric_limits<uint32_t>::max()) {
            fail_graph("fbin row stride exceeds uint32 range for " + path);
        }
        info.scalar_type = DenseVectorScalarType::Float32;
        info.rows = (row_limit > 0 && row_limit < header.rows) ? row_limit : header.rows;
        info.cols = header.cols;
        info.row_data_offset = kFbinHeaderBytes;
        info.row_stride_bytes = static_cast<uint32_t>(row_bytes);
        info.row_prefix_bytes = 0;
        return info;
    }
    if (ext == ".ibin" || ext == ".u8bin") {
        uint32_t rows = 0;
        uint32_t cols = 0;
        input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        if (!input || cols == 0) {
            fail_graph("invalid vector header in " + path);
        }
        info.scalar_type = DenseVectorScalarType::Uint8;
        info.rows = (row_limit > 0 && row_limit < rows) ? row_limit : rows;
        info.cols = cols;
        info.row_data_offset = sizeof(uint32_t) * 2ULL;
        info.row_stride_bytes = cols;
        info.row_prefix_bytes = 0;
        return info;
    }

    if (ext == ".bvecs") {
        uint32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!input || dim == 0) {
            fail_graph("invalid bvecs dimension in " + path);
        }
        input.seekg(0, std::ios::end);
        const std::streamoff file_size = input.tellg();
        const std::streamoff record_size = sizeof(uint32_t) + static_cast<std::streamoff>(dim);
        if (record_size <= 0 || file_size % record_size != 0) {
            fail_graph("bvecs file size is not row-aligned for " + path);
        }
        const uint32_t total_rows = static_cast<uint32_t>(file_size / record_size);
        info.scalar_type = DenseVectorScalarType::Uint8;
        info.rows = (row_limit > 0 && row_limit < total_rows) ? row_limit : total_rows;
        info.cols = dim;
        info.row_data_offset = sizeof(uint32_t);
        info.row_stride_bytes = static_cast<uint32_t>(record_size);
        info.row_prefix_bytes = sizeof(uint32_t);
        return info;
    }

    if (ext == ".fvecs" || ext == ".hvecs") {
        uint32_t dim = 0;
        input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        if (!input || dim == 0) {
            fail_graph("invalid " + ext + " dimension in " + path);
        }
        input.seekg(0, std::ios::end);
        const std::streamoff file_size = input.tellg();
        const std::streamoff record_size =
            sizeof(uint32_t) + static_cast<std::streamoff>(dim) *
                static_cast<std::streamoff>(ext == ".hvecs" ? sizeof(uint16_t) : sizeof(float));
        if (record_size <= 0 || file_size % record_size != 0) {
            fail_graph(ext + " file size is not row-aligned for " + path);
        }
        const uint32_t total_rows = static_cast<uint32_t>(file_size / record_size);
        info.scalar_type = ext == ".hvecs" ? DenseVectorScalarType::Float16 : DenseVectorScalarType::Float32;
        info.rows = (row_limit > 0 && row_limit < total_rows) ? row_limit : total_rows;
        info.cols = dim;
        info.row_data_offset = sizeof(uint32_t);
        info.row_stride_bytes = static_cast<uint32_t>(record_size);
        info.row_prefix_bytes = sizeof(uint32_t);
        return info;
    }
    fail_graph("unsupported vector file extension for " + path +
                 " (supported: .bvecs, .ibin, .u8bin, .fbin, .fvecs, .hvecs)");
}

void validate_vector_file_metadata(const VectorFileInfo& file_info,
                                   const DenseVectorMetadata& metadata)
{
    if (file_info.scalar_type != metadata.scalar_type ||
        file_info.rows != metadata.rows ||
        file_info.cols != metadata.cols) {
        fail_graph("vector file header does not match inspected dense vector metadata for " +
                     metadata.path);
    }
}

void require_external_base_payload_not_loaded(const GraphRuntime& runtime,
                                              bool external_base_payload)
{
    if (!external_base_payload) {
        return;
    }
    if (!runtime.base_vectors.data.empty() || !runtime.base_float_vectors.data.empty()) {
        fail_graph("external rerank vectors must not materialize the full base vector payload");
    }
}

VectorFileInfo inspect_uint8_vector_file(const std::string& path, uint32_t row_limit)
{
    VectorFileInfo info = inspect_vector_file(path, row_limit);
    if (info.scalar_type != DenseVectorScalarType::Uint8) {
        fail_graph("vector file reader requires uint8 payloads for " + path);
    }
    return info;
}

void require_uint8_vector_file(const DenseVectorMetadata& metadata,
                               const std::string& context)
{
    if (metadata.scalar_type == DenseVectorScalarType::Uint8) {
        return;
    }
    fail_graph(context + " requires uint8 dense vectors (.bvecs/.ibin/.u8bin); got " +
                 dense_vector_scalar_type_name(metadata.scalar_type) + " vectors at " +
                 metadata.path + " with shape " + std::to_string(metadata.rows) + "x" +
                 std::to_string(metadata.cols));
}

void require_uint8_graph_runtime(const GraphRuntime& runtime,
                                   const std::string& context)
{
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        return;
    }
    fail_graph(context + " requires uint8 dense vectors for the current CUDA/graph path; got base=" +
                 dense_vector_scalar_type_name(runtime.base_vector_metadata.scalar_type) +
                 " shape " + std::to_string(runtime.base_vector_metadata.rows) + "x" +
                 std::to_string(runtime.base_vector_metadata.cols) + " and query=" +
                 dense_vector_scalar_type_name(runtime.query_vector_metadata.scalar_type) +
                 " shape " + std::to_string(runtime.query_vector_metadata.rows) + "x" +
                 std::to_string(runtime.query_vector_metadata.cols) +
                 ". Float .fbin/.fvecs/.hvecs loading is supported for shape/type validation only in this path.");
}

namespace {

constexpr std::array<char, 8> kInt8RerankMagic{
    'F', 'A', 'R', 'Q', 'I', '8', '0', '1'};
constexpr uint32_t kInt8RerankVersion = 1;
constexpr uint32_t kInt8RerankHeaderBytes = 4096;
constexpr size_t kInt8BuildChunkRows = 4096;

struct Int8RerankArtifactHeader {
    std::array<char, 8> magic{};
    uint32_t version = 0;
    uint32_t header_bytes = 0;
    uint32_t source_scalar_type = 0;
    uint32_t cols = 0;
    uint64_t rows = 0;
    uint64_t codes_offset = 0;
    uint64_t scales_offset = 0;
    uint64_t norms_offset = 0;
    uint64_t file_bytes = 0;
    uint64_t source_file_bytes = 0;
    std::array<uint64_t, 7> reserved{};
};

static_assert(sizeof(Int8RerankArtifactHeader) == 128,
              "INT8 rerank artifact header identity changed");

uint64_t checked_u64_product(uint64_t lhs, uint64_t rhs, const char* description)
{
    if (rhs != 0 && lhs > std::numeric_limits<uint64_t>::max() / rhs) {
        fail_graph(std::string(description) + " overflows uint64_t");
    }
    return lhs * rhs;
}

uint64_t checked_u64_sum(uint64_t lhs, uint64_t rhs, const char* description)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        fail_graph(std::string(description) + " overflows uint64_t");
    }
    return lhs + rhs;
}

uint64_t align_u64(uint64_t value, uint64_t alignment, const char* description)
{
    if (alignment == 0) {
        fail_graph(std::string(description) + " has zero alignment");
    }
    const uint64_t remainder = value % alignment;
    return remainder == 0
        ? value
        : checked_u64_sum(value, alignment - remainder, description);
}

size_t checked_size(uint64_t value, const char* description)
{
    if (value > std::numeric_limits<size_t>::max()) {
        fail_graph(std::string(description) + " exceeds size_t");
    }
    return static_cast<size_t>(value);
}

void write_exact_at(int fd,
                    const void* source,
                    size_t bytes,
                    uint64_t offset,
                    const std::string& description)
{
    const uint8_t* input = static_cast<const uint8_t*>(source);
    size_t done = 0;
    while (done < bytes) {
        const size_t request = std::min<size_t>(bytes - done, 1ULL << 30);
        ssize_t written = -1;
        do {
            written = pwrite(fd,
                             input + done,
                             request,
                             static_cast<off_t>(offset + done));
        } while (written < 0 && errno == EINTR);
        if (written <= 0) {
            fail_graph("failed to write " + description + ": " +
                         std::strerror(errno));
        }
        done += static_cast<size_t>(written);
    }
}

void read_large_exact_at(int fd,
                         void* destination,
                         size_t bytes,
                         uint64_t offset,
                         const std::string& description)
{
    uint8_t* output = static_cast<uint8_t*>(destination);
    size_t done = 0;
    while (done < bytes) {
        const size_t request = std::min<size_t>(bytes - done, 1ULL << 30);
        ssize_t read_size = -1;
        do {
            read_size = pread(fd,
                              output + done,
                              request,
                              static_cast<off_t>(offset + done));
        } while (read_size < 0 && errno == EINTR);
        if (read_size <= 0) {
            fail_graph("failed to read " + description + ": " +
                         std::strerror(errno));
        }
        done += static_cast<size_t>(read_size);
    }
}

Int8RerankArtifactHeader make_int8_artifact_header(const VectorFileInfo& info,
                                                    uint64_t source_file_bytes)
{
    const uint64_t codes_bytes = checked_u64_product(info.rows,
                                                     info.cols,
                                                     "INT8 code payload");
    const uint64_t metadata_bytes = checked_u64_product(info.rows,
                                                        sizeof(float),
                                                        "INT8 row metadata");
    Int8RerankArtifactHeader header;
    header.magic = kInt8RerankMagic;
    header.version = kInt8RerankVersion;
    header.header_bytes = kInt8RerankHeaderBytes;
    header.source_scalar_type = static_cast<uint32_t>(info.scalar_type);
    header.cols = info.cols;
    header.rows = info.rows;
    header.codes_offset = kInt8RerankHeaderBytes;
    header.scales_offset = align_u64(
        checked_u64_sum(header.codes_offset, codes_bytes, "INT8 scale offset"),
        kInt8RerankHeaderBytes,
        "INT8 scale offset");
    header.norms_offset = align_u64(
        checked_u64_sum(header.scales_offset, metadata_bytes, "INT8 norm offset"),
        kInt8RerankHeaderBytes,
        "INT8 norm offset");
    header.file_bytes = checked_u64_sum(header.norms_offset,
                                        metadata_bytes,
                                        "INT8 artifact size");
    header.source_file_bytes = source_file_bytes;
    return header;
}

Int8RerankArtifactHeader read_int8_artifact_header(int fd,
                                                   const std::string& path)
{
    Int8RerankArtifactHeader header;
    read_large_exact_at(fd, &header, sizeof(header), 0, "INT8 artifact header " + path);
    if (header.magic != kInt8RerankMagic ||
        header.version != kInt8RerankVersion ||
        header.header_bytes != kInt8RerankHeaderBytes ||
        header.rows == 0 || header.rows > UINT32_MAX || header.cols == 0 ||
        (header.source_scalar_type != static_cast<uint32_t>(DenseVectorScalarType::Float16) &&
         header.source_scalar_type != static_cast<uint32_t>(DenseVectorScalarType::Float32))) {
        fail_graph("invalid or unsupported INT8 rerank artifact header in " + path);
    }
    VectorFileInfo shape;
    shape.rows = static_cast<uint32_t>(header.rows);
    shape.cols = header.cols;
    shape.scalar_type = static_cast<DenseVectorScalarType>(header.source_scalar_type);
    const Int8RerankArtifactHeader expected =
        make_int8_artifact_header(shape, header.source_file_bytes);
    if (header.codes_offset != expected.codes_offset ||
        header.scales_offset != expected.scales_offset ||
        header.norms_offset != expected.norms_offset ||
        header.file_bytes != expected.file_bytes) {
        fail_graph("INT8 rerank artifact offsets do not match its shape in " + path);
    }
    struct stat status {};
    if (fstat(fd, &status) != 0 || status.st_size < 0 ||
        static_cast<uint64_t>(status.st_size) != header.file_bytes) {
        fail_graph("INT8 rerank artifact file size does not match its header in " + path);
    }
    return header;
}

}  // namespace

namespace {

enum class Int8DistanceKernel : uint8_t {
    ScalarL2,
    Avx512Dot,
};

float int8_l2_scalar(const int8_t* code,
                     const float* query,
                     uint32_t cols,
                     float scale) noexcept
{
    float distance = 0.0f;
    for (uint32_t col = 0; col < cols; ++col) {
        const float diff = query[col] - scale * static_cast<float>(code[col]);
        distance += diff * diff;
    }
    return distance;
}

#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
float horizontal_sum_avx512(__m512 value) noexcept
{
    alignas(64) float lanes[16];
    _mm512_store_ps(lanes, value);
    float sum = 0.0f;
    for (float lane : lanes) {
        sum += lane;
    }
    return sum;
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
__m512 load_int8_as_float_avx512(const int8_t* values) noexcept
{
    const __m128i packed = _mm_loadu_si128(
        reinterpret_cast<const __m128i*>(values));
    const __m512i expanded = _mm512_maskz_cvtepi8_epi32(0xffff, packed);
    return _mm512_maskz_cvtepi32_ps(0xffff, expanded);
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
float squared_norm_avx512(const float* values, uint32_t cols) noexcept
{
    __m512 sum0 = _mm512_setzero_ps();
    __m512 sum1 = _mm512_setzero_ps();
    __m512 sum2 = _mm512_setzero_ps();
    __m512 sum3 = _mm512_setzero_ps();
    uint32_t col = 0;
    for (; col + 64 <= cols; col += 64) {
        const __m512 value0 = _mm512_loadu_ps(values + col);
        const __m512 value1 = _mm512_loadu_ps(values + col + 16);
        const __m512 value2 = _mm512_loadu_ps(values + col + 32);
        const __m512 value3 = _mm512_loadu_ps(values + col + 48);
        sum0 = _mm512_fmadd_ps(value0, value0, sum0);
        sum1 = _mm512_fmadd_ps(value1, value1, sum1);
        sum2 = _mm512_fmadd_ps(value2, value2, sum2);
        sum3 = _mm512_fmadd_ps(value3, value3, sum3);
    }
    for (; col + 16 <= cols; col += 16) {
        const __m512 value = _mm512_loadu_ps(values + col);
        sum0 = _mm512_fmadd_ps(value, value, sum0);
    }
    const __m512 sum01 = _mm512_add_ps(sum0, sum1);
    const __m512 sum23 = _mm512_add_ps(sum2, sum3);
    float squared_norm = horizontal_sum_avx512(_mm512_add_ps(sum01, sum23));
    for (; col < cols; ++col) {
        squared_norm += values[col] * values[col];
    }
    return squared_norm;
}

__attribute__((target("avx512f,avx512bw,avx512vl,fma")))
float int8_l2_avx512_dot(const int8_t* code,
                         const float* query,
                         uint32_t cols,
                         float scale,
                         float query_squared_norm,
                         float row_squared_norm) noexcept
{
    __m512 dot0 = _mm512_setzero_ps();
    __m512 dot1 = _mm512_setzero_ps();
    __m512 dot2 = _mm512_setzero_ps();
    __m512 dot3 = _mm512_setzero_ps();
    uint32_t col = 0;
    for (; col + 64 <= cols; col += 64) {
        const __m512 code0 = load_int8_as_float_avx512(code + col);
        const __m512 code1 = load_int8_as_float_avx512(code + col + 16);
        const __m512 code2 = load_int8_as_float_avx512(code + col + 32);
        const __m512 code3 = load_int8_as_float_avx512(code + col + 48);
        dot0 = _mm512_fmadd_ps(_mm512_loadu_ps(query + col), code0, dot0);
        dot1 = _mm512_fmadd_ps(_mm512_loadu_ps(query + col + 16), code1, dot1);
        dot2 = _mm512_fmadd_ps(_mm512_loadu_ps(query + col + 32), code2, dot2);
        dot3 = _mm512_fmadd_ps(_mm512_loadu_ps(query + col + 48), code3, dot3);
    }
    for (; col + 16 <= cols; col += 16) {
        const __m512 code_values = load_int8_as_float_avx512(code + col);
        dot0 = _mm512_fmadd_ps(_mm512_loadu_ps(query + col), code_values, dot0);
    }
    const __m512 dot01 = _mm512_add_ps(dot0, dot1);
    const __m512 dot23 = _mm512_add_ps(dot2, dot3);
    float dot = horizontal_sum_avx512(_mm512_add_ps(dot01, dot23));
    for (; col < cols; ++col) {
        dot += query[col] * static_cast<float>(code[col]);
    }
    const float distance = std::fma(-2.0f * scale,
                                    dot,
                                    query_squared_norm + row_squared_norm);
    return std::max(distance, 0.0f);
}

bool cpu_supports_int8_avx512_kernel() noexcept
{
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") &&
           __builtin_cpu_supports("avx512bw") &&
           __builtin_cpu_supports("avx512vl") &&
           __builtin_cpu_supports("fma");
}

#else

bool cpu_supports_int8_avx512_kernel() noexcept
{
    return false;
}

#endif

}  // namespace

struct RerankVectorStore::Impl {
    enum class Kind : uint8_t {
        MmapFp16,
        MemoryInt8,
    };

    ~Impl()
    {
        if (mapping != MAP_FAILED && mapping_bytes != 0) {
            munmap(mapping, mapping_bytes);
        }
        if (fd >= 0) {
            close(fd);
        }
    }

    void validate_row(uint32_t row, uint32_t requested_cols) const
    {
        if (row >= rows) {
            fail_graph("rerank vector row out of range");
        }
        if (requested_cols != cols) {
            fail_graph("rerank vector dimension mismatch");
        }
    }

    const uint16_t* fp16_row(uint32_t row) const
    {
        const uint64_t record_offset = checked_u64_product(
            row, source_info.row_stride_bytes, "mmap FP16 row offset");
        const uint64_t payload_offset = checked_u64_sum(
            record_offset, source_info.row_prefix_bytes, "mmap FP16 payload offset");
        uint32_t row_dim = 0;
        std::memcpy(&row_dim,
                    static_cast<const uint8_t*>(mapping) + record_offset,
                    sizeof(row_dim));
        if (row_dim != cols) {
            fail_graph("mmap FP16 row dimension mismatch at row " +
                         std::to_string(row));
        }
        return reinterpret_cast<const uint16_t*>(
            static_cast<const uint8_t*>(mapping) + payload_offset);
    }

    Kind kind = Kind::MmapFp16;
    uint32_t rows = 0;
    uint32_t cols = 0;
    int fd = -1;
    void* mapping = MAP_FAILED;
    size_t mapping_bytes = 0;
    uint64_t resident_bytes = 0;
    VectorFileInfo source_info;
    const int8_t* codes = nullptr;
    const float* scales = nullptr;
    const float* norms = nullptr;
    Int8DistanceKernel int8_distance_kernel = Int8DistanceKernel::ScalarL2;
    go_fann::fp16_l2::Kernel fp16_kernel = go_fann::fp16_l2::Kernel::Scalar;
    size_t fp16_batch_width = 1;
    size_t fp16_prefetch_rows = 0;
    std::string fp16_kernel_name;
};

RerankVectorStore::RerankVectorStore(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl))
{
}

RerankVectorStore::~RerankVectorStore() = default;

std::shared_ptr<RerankVectorStore> RerankVectorStore::map_fp16(
    const VectorFileInfo& source_info)
{
    if (source_info.scalar_type != DenseVectorScalarType::Float16 ||
        source_info.rows == 0 || source_info.cols == 0 ||
        source_info.row_data_offset != sizeof(uint32_t) ||
        source_info.row_prefix_bytes != sizeof(uint32_t) ||
        source_info.row_stride_bytes != sizeof(uint32_t) +
            static_cast<uint64_t>(source_info.cols) * sizeof(uint16_t)) {
        fail_graph("mmap_fp16 requires a strided .hvecs source");
    }
    auto impl = std::make_unique<Impl>();
    impl->kind = Impl::Kind::MmapFp16;
    const char* kernel_env = std::getenv("GO_FANN_FP16_RERANK_KERNEL");
    impl->fp16_kernel = go_fann::fp16_l2::select(kernel_env ? kernel_env : "auto");
    const auto read_control = [](const char* key, size_t fallback, size_t maximum) {
        const char* value = std::getenv(key);
        if (!value || !*value) return fallback;
        const std::string input(value);
        if (input.find_first_not_of("0123456789") != std::string::npos)
            fail_graph(std::string(key) + " must be an unsigned integer");
        size_t used = 0;
        const unsigned long long parsed = std::stoull(input, &used);
        if (used != input.size() || parsed > maximum)
            fail_graph(std::string(key) + " is out of range");
        return static_cast<size_t>(parsed);
    };
    impl->fp16_batch_width = read_control("GO_FANN_FP16_RERANK_BATCH", 4, 8);
    if (impl->fp16_batch_width != 1 && impl->fp16_batch_width != 4 && impl->fp16_batch_width != 8)
        fail_graph("GO_FANN_FP16_RERANK_BATCH must be 1, 4 or 8");
    impl->fp16_prefetch_rows = read_control("GO_FANN_FP16_RERANK_PREFETCH", 0, 64);
    if (impl->fp16_kernel == go_fann::fp16_l2::Kernel::Scalar) {
        impl->fp16_batch_width = 1;
        impl->fp16_prefetch_rows = 0;
    }
    impl->fp16_kernel_name = std::string(go_fann::fp16_l2::name(impl->fp16_kernel)) +
        "_batch" + std::to_string(impl->fp16_batch_width) +
        "_prefetch" + std::to_string(impl->fp16_prefetch_rows);
    impl->rows = source_info.rows;
    impl->cols = source_info.cols;
    impl->source_info = source_info;
    impl->fd = open(source_info.path.c_str(), O_RDONLY | O_CLOEXEC);
    if (impl->fd < 0) {
        fail_graph("unable to open mmap FP16 source " + source_info.path + ": " +
                     std::strerror(errno));
    }
    struct stat status {};
    if (fstat(impl->fd, &status) != 0 || status.st_size <= 0) {
        fail_graph("unable to stat mmap FP16 source " + source_info.path);
    }
    impl->mapping_bytes = checked_size(static_cast<uint64_t>(status.st_size),
                                       "mmap FP16 file size");
    const uint64_t required_bytes = checked_u64_product(
        source_info.rows, source_info.row_stride_bytes, "mmap FP16 required file size");
    if (required_bytes > impl->mapping_bytes ||
        impl->mapping_bytes % source_info.row_stride_bytes != 0) {
        fail_graph("mmap FP16 file size does not match vector metadata");
    }
    impl->mapping = mmap(nullptr,
                         impl->mapping_bytes,
                         PROT_READ,
                         MAP_SHARED,
                         impl->fd,
                         0);
    if (impl->mapping == MAP_FAILED) {
        fail_graph("unable to mmap FP16 source " + source_info.path + ": " +
                     std::strerror(errno));
    }
    (void)madvise(impl->mapping, impl->mapping_bytes, MADV_RANDOM);
    (void)madvise(impl->mapping, impl->mapping_bytes, MADV_DONTDUMP);
    return std::shared_ptr<RerankVectorStore>(
        new RerankVectorStore(std::move(impl)));
}

std::shared_ptr<RerankVectorStore> RerankVectorStore::load_memory_int8(
    const std::string& artifact_path,
    const DenseVectorMetadata& source_metadata)
{
    int fd = open(artifact_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        fail_graph("unable to open INT8 rerank artifact " + artifact_path + ": " +
                     std::strerror(errno));
    }
    try {
        const Int8RerankArtifactHeader header = read_int8_artifact_header(fd, artifact_path);
        if (header.rows != source_metadata.rows ||
            header.cols != source_metadata.cols ||
            header.source_scalar_type != static_cast<uint32_t>(source_metadata.scalar_type)) {
            fail_graph("INT8 rerank artifact shape/type does not match base vectors");
        }
        std::error_code source_size_error;
        const uint64_t source_file_bytes = std::filesystem::file_size(
            source_metadata.path, source_size_error);
        if (source_size_error || source_file_bytes != header.source_file_bytes) {
            fail_graph("INT8 rerank artifact source file identity does not match base vectors");
        }
        const uint64_t codes_bytes_u64 = checked_u64_product(
            header.rows, header.cols, "resident INT8 codes");
        const uint64_t metadata_bytes_u64 = checked_u64_product(
            header.rows, sizeof(float), "resident INT8 metadata");
        const uint64_t scales_memory_offset = align_u64(codes_bytes_u64,
                                                       64,
                                                       "resident INT8 scales");
        const uint64_t norms_memory_offset = align_u64(
            checked_u64_sum(scales_memory_offset,
                            metadata_bytes_u64,
                            "resident INT8 norms"),
            64,
            "resident INT8 norms");
        const uint64_t resident_bytes_u64 = checked_u64_sum(
            norms_memory_offset, metadata_bytes_u64, "resident INT8 payload");
        const size_t resident_bytes = checked_size(resident_bytes_u64,
                                                   "resident INT8 payload");
        void* mapping = mmap(nullptr,
                             resident_bytes,
                             PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS,
                             -1,
                             0);
        if (mapping == MAP_FAILED) {
            fail_graph("unable to allocate resident INT8 rerank payload: " +
                         std::string(std::strerror(errno)));
        }
        try {
            uint8_t* base = static_cast<uint8_t*>(mapping);
            read_large_exact_at(fd,
                                base,
                                checked_size(codes_bytes_u64, "resident INT8 codes"),
                                header.codes_offset,
                                "INT8 codes");
            read_large_exact_at(fd,
                                base + scales_memory_offset,
                                checked_size(metadata_bytes_u64, "resident INT8 scales"),
                                header.scales_offset,
                                "INT8 scales");
            read_large_exact_at(fd,
                                base + norms_memory_offset,
                                checked_size(metadata_bytes_u64, "resident INT8 norms"),
                                header.norms_offset,
                                "INT8 norms");
        } catch (...) {
            munmap(mapping, resident_bytes);
            throw;
        }
        (void)madvise(mapping, resident_bytes, MADV_HUGEPAGE);
        (void)madvise(mapping, resident_bytes, MADV_DONTDUMP);
        if (mprotect(mapping, resident_bytes, PROT_READ) != 0) {
            munmap(mapping, resident_bytes);
            fail_graph("unable to protect resident INT8 rerank payload: " +
                         std::string(std::strerror(errno)));
        }
        (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
        close(fd);
        fd = -1;

        auto impl = std::make_unique<Impl>();
        impl->kind = Impl::Kind::MemoryInt8;
        impl->rows = static_cast<uint32_t>(header.rows);
        impl->cols = header.cols;
        impl->mapping = mapping;
        impl->mapping_bytes = resident_bytes;
        impl->resident_bytes = resident_bytes;
        const uint8_t* base = static_cast<const uint8_t*>(mapping);
        impl->codes = reinterpret_cast<const int8_t*>(base);
        impl->scales = reinterpret_cast<const float*>(base + scales_memory_offset);
        impl->norms = reinterpret_cast<const float*>(base + norms_memory_offset);
        impl->int8_distance_kernel = cpu_supports_int8_avx512_kernel()
            ? Int8DistanceKernel::Avx512Dot
            : Int8DistanceKernel::ScalarL2;
        return std::shared_ptr<RerankVectorStore>(
            new RerankVectorStore(std::move(impl)));
    } catch (...) {
        if (fd >= 0) {
            close(fd);
        }
        throw;
    }
}

uint32_t RerankVectorStore::rows() const noexcept
{
    return impl_->rows;
}

uint32_t RerankVectorStore::cols() const noexcept
{
    return impl_->cols;
}

std::string_view RerankVectorStore::mode() const noexcept
{
    return impl_->kind == Impl::Kind::MmapFp16 ? "mmap_fp16" : "memory_int8";
}

std::string_view RerankVectorStore::distance_kernel() const noexcept
{
    if (impl_->kind == Impl::Kind::MmapFp16) {
        return impl_->fp16_kernel_name;
    }
    return impl_->int8_distance_kernel == Int8DistanceKernel::Avx512Dot
        ? "int8_avx512_dot"
        : "int8_scalar_l2";
}

uint64_t RerankVectorStore::mapped_bytes() const noexcept
{
    return impl_->kind == Impl::Kind::MmapFp16 ? impl_->mapping_bytes : 0;
}

uint64_t RerankVectorStore::resident_payload_bytes() const noexcept
{
    return impl_->resident_bytes;
}

PreparedFloatL2Query RerankVectorStore::prepare_l2_query(
    const float* query,
    uint32_t cols) const
{
    if (query == nullptr) {
        fail_graph("rerank query vector is null");
    }
    if (cols != impl_->cols) {
        fail_graph("rerank query dimension mismatch");
    }
    PreparedFloatL2Query prepared;
    prepared.values = query;
    prepared.cols = cols;
    if (impl_->kind == Impl::Kind::MemoryInt8 &&
        impl_->int8_distance_kernel == Int8DistanceKernel::Avx512Dot) {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        prepared.squared_norm = squared_norm_avx512(query, cols);
#endif
    }
    return prepared;
}

void RerankVectorStore::read_float_row(uint32_t row,
                                       float* dst,
                                       uint32_t cols) const
{
    impl_->validate_row(row, cols);
    if (dst == nullptr) {
        fail_graph("rerank vector destination is null");
    }
    if (impl_->kind == Impl::Kind::MmapFp16) {
        convert_half_row_to_float(impl_->fp16_row(row), dst, cols);
        return;
    }
    const int8_t* code = impl_->codes + static_cast<size_t>(row) * cols;
    const float scale = impl_->scales[row];
    for (uint32_t col = 0; col < cols; ++col) {
        dst[col] = scale * static_cast<float>(code[col]);
    }
}

float RerankVectorStore::l2_float(uint32_t row,
                                  const PreparedFloatL2Query& query) const
{
    impl_->validate_row(row, query.cols);
    if (query.values == nullptr) {
        fail_graph("rerank query vector is null");
    }
    if (impl_->kind == Impl::Kind::MmapFp16) {
        const uint16_t* base = impl_->fp16_row(row);
        float distance;
        go_fann::fp16_l2::score<1>(impl_->fp16_kernel, &base,
                                        query.values, query.cols, &distance);
        return distance;
    }
    const int8_t* code = impl_->codes + static_cast<size_t>(row) * query.cols;
    const float scale = impl_->scales[row];
    if (impl_->int8_distance_kernel == Int8DistanceKernel::Avx512Dot) {
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
        return int8_l2_avx512_dot(code,
                                 query.values,
                                 query.cols,
                                 scale,
                                 query.squared_norm,
                                 impl_->norms[row]);
#endif
    }
    return int8_l2_scalar(code, query.values, query.cols, scale);
}

float RerankVectorStore::l2_float(uint32_t row,
                                  const float* query,
                                  uint32_t cols) const
{
    return l2_float(row, prepare_l2_query(query, cols));
}

bool RerankVectorStore::uses_batched_l2() const noexcept
{
    return impl_->kind == Impl::Kind::MmapFp16 && impl_->fp16_batch_width > 1;
}

void RerankVectorStore::l2_float_batch(const uint32_t* rows, size_t count,
                                      const PreparedFloatL2Query& query,
                                      float* distances) const
{
    if (count == 0) return;
    if (!rows || !distances || !query.values || query.cols != impl_->cols)
        fail_graph("invalid batched rerank buffers or query dimension");
    if (impl_->kind != Impl::Kind::MmapFp16) {
        for (size_t i = 0; i < count; ++i) distances[i] = l2_float(rows[i], query);
        return;
    }
    for (size_t begin = 0; begin < count; begin += impl_->fp16_batch_width) {
        const size_t n = std::min(impl_->fp16_batch_width, count - begin);
        std::array<const uint16_t*, 8> pointers{};
        for (size_t i = 0; i < n; ++i) {
            impl_->validate_row(rows[begin + i], query.cols);
            pointers[i] = impl_->fp16_row(rows[begin + i]);
#if GO_FANN_FP16_X86
            const size_t ahead = begin + i + impl_->fp16_prefetch_rows;
            if (impl_->fp16_prefetch_rows != 0 && ahead < count) {
                impl_->validate_row(rows[ahead], query.cols);
                const auto* payload = static_cast<const uint8_t*>(impl_->mapping) +
                    static_cast<uint64_t>(rows[ahead]) * impl_->source_info.row_stride_bytes +
                    impl_->source_info.row_prefix_bytes;
                for (size_t byte = 0; byte < static_cast<size_t>(query.cols) * 2; byte += 64)
                    __builtin_prefetch(payload + byte, 0, 1);
            }
#endif
        }
        go_fann::fp16_l2::batch(impl_->fp16_kernel, pointers.data(), n,
                                     query.values, query.cols, distances + begin,
                                     impl_->fp16_batch_width);
    }
}

Int8RerankBuildStats build_int8_rerank_artifact(const std::string& source_path,
                                                 const std::string& artifact_path,
                                                 uint32_t row_limit,
                                                 bool force_rebuild)
{
    if (source_path.empty() || artifact_path.empty()) {
        fail_graph("INT8 rerank build requires source and artifact paths");
    }
    const std::filesystem::path normalized_source =
        std::filesystem::absolute(source_path).lexically_normal();
    const std::filesystem::path normalized_artifact =
        std::filesystem::absolute(artifact_path).lexically_normal();
    if (normalized_source == normalized_artifact) {
        fail_graph("INT8 rerank source and artifact paths must differ");
    }
    if (!force_rebuild && std::filesystem::exists(artifact_path)) {
        fail_graph("INT8 rerank artifact already exists: " + artifact_path);
    }
    const VectorFileInfo info = inspect_vector_file(source_path, row_limit);
    if (!dense_vector_scalar_is_float(info.scalar_type) ||
        info.row_prefix_bytes != sizeof(uint32_t)) {
        fail_graph("INT8 rerank builder requires .fbin, .fvecs, or .hvecs source vectors");
    }
    const size_t expected_stride = sizeof(uint32_t) +
        static_cast<size_t>(info.cols) * dense_vector_scalar_size(info.scalar_type);
    if (info.row_stride_bytes != expected_stride) {
        fail_graph("INT8 rerank source row stride is incompatible with its shape");
    }

    int source_fd = open(source_path.c_str(), O_RDONLY | O_CLOEXEC);
    if (source_fd < 0) {
        fail_graph("unable to open INT8 rerank source " + source_path + ": " +
                     std::strerror(errno));
    }
    const std::string temporary_path = artifact_path + ".tmp." +
        std::to_string(static_cast<unsigned long long>(getpid()));
    int output_fd = -1;
    try {
        struct stat source_status {};
        if (fstat(source_fd, &source_status) != 0 || source_status.st_size < 0) {
            fail_graph("unable to stat INT8 rerank source " + source_path);
        }
        const Int8RerankArtifactHeader header = make_int8_artifact_header(
            info, static_cast<uint64_t>(source_status.st_size));
        if (header.file_bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            fail_graph("INT8 rerank artifact exceeds off_t range");
        }
        output_fd = open(temporary_path.c_str(),
                         O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC,
                         0644);
        if (output_fd < 0) {
            fail_graph("unable to create temporary INT8 rerank artifact " +
                         temporary_path + ": " + std::strerror(errno));
        }
        if (ftruncate(output_fd, static_cast<off_t>(header.file_bytes)) != 0) {
            fail_graph("unable to size temporary INT8 rerank artifact: " +
                         std::string(std::strerror(errno)));
        }
        std::array<uint8_t, kInt8RerankHeaderBytes> header_block{};
        std::memcpy(header_block.data(), &header, sizeof(header));
        write_exact_at(output_fd,
                       header_block.data(),
                       header_block.size(),
                       0,
                       "INT8 rerank artifact header");

        const size_t chunk_rows = std::min<size_t>(kInt8BuildChunkRows, info.rows);
        std::vector<uint8_t> source_chunk(chunk_rows * expected_stride);
        std::vector<int8_t> code_chunk(chunk_rows * info.cols);
        std::vector<float> scale_chunk(chunk_rows);
        std::vector<float> norm_chunk(chunk_rows);
        uint64_t next_progress = std::max<uint64_t>(1, info.rows / 100);
        uint64_t progress_threshold = next_progress;
        for (uint64_t begin = 0; begin < info.rows; begin += chunk_rows) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(chunk_rows, info.rows - begin));
            const size_t source_bytes = count * expected_stride;
            read_large_exact_at(source_fd,
                                source_chunk.data(),
                                source_bytes,
                                begin * expected_stride,
                                "INT8 rerank source chunk");
            std::atomic<uint64_t> first_invalid{UINT64_MAX};
#pragma omp parallel for schedule(static)
            for (int64_t local_signed = 0;
                 local_signed < static_cast<int64_t>(count);
                 ++local_signed) {
                const size_t local = static_cast<size_t>(local_signed);
                const uint8_t* record = source_chunk.data() + local * expected_stride;
                uint32_t row_dim = 0;
                std::memcpy(&row_dim, record, sizeof(row_dim));
                if (row_dim != info.cols) {
                    uint64_t expected = UINT64_MAX;
                    first_invalid.compare_exchange_strong(expected, begin + local);
                    continue;
                }
                const uint8_t* payload = record + sizeof(uint32_t);
                float max_abs = 0.0f;
                bool finite = true;
                for (uint32_t col = 0; col < info.cols; ++col) {
                    float value = 0.0f;
                    if (info.scalar_type == DenseVectorScalarType::Float16) {
                        uint16_t half = 0;
                        std::memcpy(&half, payload + static_cast<size_t>(col) * sizeof(half), sizeof(half));
                        value = half_to_float(half);
                    } else {
                        std::memcpy(&value,
                                    payload + static_cast<size_t>(col) * sizeof(value),
                                    sizeof(value));
                    }
                    if (!std::isfinite(value)) {
                        finite = false;
                        break;
                    }
                    max_abs = std::max(max_abs, std::abs(value));
                }
                if (!finite) {
                    uint64_t expected = UINT64_MAX;
                    first_invalid.compare_exchange_strong(expected, begin + local);
                    continue;
                }
                const float scale = max_abs == 0.0f ? 0.0f : max_abs / 127.0f;
                double norm = 0.0;
                int8_t* row_codes = code_chunk.data() + local * info.cols;
                for (uint32_t col = 0; col < info.cols; ++col) {
                    float value = 0.0f;
                    if (info.scalar_type == DenseVectorScalarType::Float16) {
                        uint16_t half = 0;
                        std::memcpy(&half, payload + static_cast<size_t>(col) * sizeof(half), sizeof(half));
                        value = half_to_float(half);
                    } else {
                        std::memcpy(&value,
                                    payload + static_cast<size_t>(col) * sizeof(value),
                                    sizeof(value));
                    }
                    const int quantized = scale == 0.0f
                        ? 0
                        : std::max(-127,
                                   std::min(127,
                                            static_cast<int>(std::round(value / scale))));
                    row_codes[col] = static_cast<int8_t>(quantized);
                    const double decoded = static_cast<double>(scale) * quantized;
                    norm += decoded * decoded;
                }
                scale_chunk[local] = scale;
                norm_chunk[local] = static_cast<float>(norm);
            }
            if (first_invalid.load() != UINT64_MAX) {
                fail_graph("invalid dimension or non-finite value in INT8 rerank source row " +
                             std::to_string(first_invalid.load()));
            }
            const uint64_t code_offset = checked_u64_sum(
                header.codes_offset,
                checked_u64_product(begin, info.cols, "INT8 code write offset"),
                "INT8 code write offset");
            const uint64_t metadata_offset = checked_u64_product(
                begin, sizeof(float), "INT8 metadata write offset");
            write_exact_at(output_fd,
                           code_chunk.data(),
                           count * info.cols,
                           code_offset,
                           "INT8 code chunk");
            write_exact_at(output_fd,
                           scale_chunk.data(),
                           count * sizeof(float),
                           header.scales_offset + metadata_offset,
                           "INT8 scale chunk");
            write_exact_at(output_fd,
                           norm_chunk.data(),
                           count * sizeof(float),
                           header.norms_offset + metadata_offset,
                           "INT8 norm chunk");
            const uint64_t completed = begin + count;
            if (completed >= progress_threshold || completed == info.rows) {
                std::cout << "[rerank-int8-build] rows=" << completed << "/" << info.rows
                          << " progress="
                          << (100.0 * static_cast<double>(completed) / info.rows)
                          << "%\n";
                std::cout.flush();
                while (progress_threshold <= completed) {
                    progress_threshold += next_progress;
                }
            }
        }
        if (fsync(output_fd) != 0) {
            fail_graph("failed to fsync INT8 rerank artifact: " +
                         std::string(std::strerror(errno)));
        }
        close(output_fd);
        output_fd = -1;
        close(source_fd);
        source_fd = -1;
        if (rename(temporary_path.c_str(), artifact_path.c_str()) != 0) {
            fail_graph("failed to publish INT8 rerank artifact: " +
                         std::string(std::strerror(errno)));
        }
        return {info.rows,
                info.cols,
                static_cast<uint64_t>(source_status.st_size),
                header.file_bytes};
    } catch (...) {
        if (output_fd >= 0) {
            close(output_fd);
        }
        if (source_fd >= 0) {
            close(source_fd);
        }
        unlink(temporary_path.c_str());
        throw;
    }
}

void copy_base_vector_row(const GraphRuntime& runtime,
                          uint32_t sample_id,
                          uint8_t* dst,
                          uint32_t cols)
{
    require_uint8_graph_runtime(runtime, "copy_base_vector_row");
    if (sample_id >= runtime.base_vectors.rows) {
        fail_graph("base vector row out of range");
    }
    if (runtime.base_vectors.cols != cols) {
        fail_graph("base vector dimension mismatch");
    }
    if (dst == nullptr || runtime.base_vectors.data.size() <
        (static_cast<uint64_t>(sample_id) + 1) * cols) {
        fail_graph("resident base vector payload is unavailable");
    }
    std::memcpy(dst, runtime.base_vectors.row_ptr(sample_id), cols);
}

void copy_lowfreq_base_vector_row(const GraphRuntime& runtime,
                                  uint32_t sample_id,
                                  uint8_t* dst,
                                  uint32_t cols)
{
    require_uint8_graph_runtime(runtime, "copy_lowfreq_base_vector_row");
    copy_base_vector_row(runtime, sample_id, dst, cols);
}

void copy_base_float_vector_row(const GraphRuntime& runtime,
                                uint32_t sample_id,
                                float* dst,
                                uint32_t cols)
{
    if (!dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        fail_graph("copy_base_float_vector_row requires float runtime vectors");
    }
    if (runtime.rerank_vector_store) {
        runtime.rerank_vector_store->read_float_row(sample_id, dst, cols);
        return;
    }
    if (sample_id >= runtime.base_float_vectors.rows) {
        fail_graph("base float vector row out of range");
    }
    if (runtime.base_float_vectors.cols != cols) {
        fail_graph("base float vector dimension mismatch");
    }
    if (dst == nullptr || runtime.base_float_vectors.data.size() <
        (static_cast<uint64_t>(sample_id) + 1) * cols) {
        fail_graph("resident base float vector payload is unavailable");
    }
    std::memcpy(dst, runtime.base_float_vectors.row_ptr(sample_id), static_cast<size_t>(cols) * sizeof(float));
}

void copy_lowfreq_base_float_vector_row(const GraphRuntime& runtime,
                                        uint32_t sample_id,
                                        float* dst,
                                        uint32_t cols)
{
    if (!dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        fail_graph("copy_lowfreq_base_float_vector_row requires float runtime vectors");
    }
    if (runtime.rerank_vector_store) {
        runtime.rerank_vector_store->read_float_row(sample_id, dst, cols);
        return;
    }
    copy_base_float_vector_row(runtime, sample_id, dst, cols);
}

PreparedFloatL2Query prepare_base_float_vector_l2_query(
    const GraphRuntime& runtime,
    const float* query,
    uint32_t cols)
{
    if (runtime.rerank_vector_store) {
        return runtime.rerank_vector_store->prepare_l2_query(query, cols);
    }
    if (query == nullptr) {
        fail_graph("base vector L2 query is null");
    }
    return PreparedFloatL2Query{query, cols, 0.0f};
}

float compute_base_float_vector_l2(const GraphRuntime& runtime,
                                   uint32_t sample_id,
                                   const PreparedFloatL2Query& query)
{
    if (!dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        fail_graph("compute_base_float_vector_l2 requires float runtime vectors");
    }
    if (runtime.rerank_vector_store) {
        return runtime.rerank_vector_store->l2_float(sample_id, query);
    }
    if (query.values == nullptr || query.cols != runtime.base_float_vectors.cols) {
        fail_graph("base vector L2 query dimension mismatch or null query");
    }
    if (sample_id >= runtime.base_float_vectors.rows) {
        fail_graph("base float vector row out of range");
    }
    if (runtime.base_float_vectors.data.size() <
        (static_cast<uint64_t>(sample_id) + 1) * query.cols) {
        fail_graph("resident base float vector payload is unavailable");
    }
    const float* base = runtime.base_float_vectors.row_ptr(sample_id);
    float distance = 0.0f;
    for (uint32_t col = 0; col < query.cols; ++col) {
        const float diff = query.values[col] - base[col];
        distance += diff * diff;
    }
    return distance;
}

GraphRouteSplitStats collect_graph_route_split_stats(const GraphRuntime& runtime,
                                                         const std::vector<int>& route_steps)
{
    GraphRouteSplitStats stats;
    const size_t route_count = runtime.query_vectors.rows;
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (route_idx < runtime.route_enabled.size() &&
            runtime.route_enabled[route_idx] == 0) {
            continue;
        }
        const int64_t steps = route_idx < route_steps.size()
            ? static_cast<int64_t>(route_steps[route_idx])
            : 0;
        if (route_idx < runtime.route_providers.size() &&
            runtime.route_providers[route_idx] != QueryRouteProvider::Label) {
            ++stats.provider_route_count;
            continue;
        }
        const bool valid_group = route_idx < runtime.query_group_ids.size() &&
                                 runtime.query_group_ids[route_idx] != UINT32_MAX;
        if (!valid_group) {
            ++stats.unroutable_route_count;
            continue;
        }
        if (route_idx < runtime.query_is_lowfreq.size() &&
            runtime.query_is_lowfreq[route_idx] != 0) {
            ++stats.lowfreq_route_count;
            stats.lowfreq_total_steps += steps;
            continue;
        }
        const bool valid_highfreq_seed = route_idx < runtime.start_local_ids.size() &&
                                         runtime.start_local_ids[route_idx] != UINT32_MAX;
        if (valid_highfreq_seed) {
            ++stats.highfreq_route_count;
            stats.highfreq_total_steps += steps;
            continue;
        }
        ++stats.unroutable_route_count;
    }
    return stats;
}

std::vector<int> build_graph_original_route_flags(const GraphRuntime& runtime)
{
    std::vector<int> flags(runtime.original_query_count, 0);
    for (size_t route_idx = 0; route_idx < runtime.route_to_original_query.size(); ++route_idx) {
        if (route_idx < runtime.route_enabled.size() &&
            runtime.route_enabled[route_idx] == 0) {
            continue;
        }
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        if (original_query >= flags.size()) {
            continue;
        }
        if (route_idx < runtime.route_providers.size() &&
            runtime.route_providers[route_idx] != QueryRouteProvider::Label) {
            flags[original_query] |= 4;
            continue;
        }
        const bool valid_group = route_idx < runtime.query_group_ids.size() &&
                                 runtime.query_group_ids[route_idx] != UINT32_MAX;
        if (!valid_group) {
            continue;
        }
        if (route_idx < runtime.query_is_lowfreq.size() &&
            runtime.query_is_lowfreq[route_idx] != 0) {
            flags[original_query] |= 2;
            continue;
        }
        const bool valid_highfreq_seed = route_idx < runtime.start_local_ids.size() &&
                                         runtime.start_local_ids[route_idx] != UINT32_MAX;
        if (valid_highfreq_seed) {
            flags[original_query] |= 1;
        }
    }
    return flags;
}

std::vector<float> load_float_fbin_column(const std::string& path,
                                          uint32_t row_limit,
                                          uint32_t* out_rows,
                                          uint32_t* out_cols)
{
    if (out_rows != nullptr) {
        *out_rows = 0;
    }
    if (out_cols != nullptr) {
        *out_cols = 0;
    }
    if (path.empty()) {
        return {};
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open float fbin file " + path);
    }

    uint32_t rows = 0;
    uint32_t cols = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!input) {
        fail_graph("invalid float fbin header in " + path);
    }

    const uint32_t effective_rows = (row_limit > 0 && row_limit < rows) ? row_limit : rows;
    std::vector<float> values(static_cast<size_t>(effective_rows) * cols, 0.0f);
    input.read(reinterpret_cast<char*>(values.data()), sizeof(float) * values.size());
    if (!input) {
        fail_graph("failed to read float fbin payload from " + path);
    }
    if (out_rows != nullptr) {
        *out_rows = effective_rows;
    }
    if (out_cols != nullptr) {
        *out_cols = cols;
    }
    return values;
}

std::string lowercase_ascii(std::string value)
{
    for (char& ch : value) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    return value;
}

std::vector<std::string> split_csv_lower_set(const std::string& value)
{
    std::vector<std::string> items;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, ',')) {
        size_t begin = 0;
        while (begin < item.size() && std::isspace(static_cast<unsigned char>(item[begin]))) {
            ++begin;
        }
        size_t end = item.size();
        while (end > begin && std::isspace(static_cast<unsigned char>(item[end - 1]))) {
            --end;
        }
        if (begin == end) {
            continue;
        }
        items.push_back(lowercase_ascii(item.substr(begin, end - begin)));
    }
    std::sort(items.begin(), items.end());
    items.erase(std::unique(items.begin(), items.end()), items.end());
    return items;
}

bool contains_sorted_string(const std::vector<std::string>& values, const std::string& value)
{
    return std::binary_search(values.begin(), values.end(), value);
}

std::string json_string_value(const json& object, const char* key)
{
    const auto it = object.find(key);
    if (it == object.end() || it->is_null()) {
        return "";
    }
    if (it->is_string()) {
        return it->get<std::string>();
    }
    if (it->is_number_integer() || it->is_number_unsigned()) {
        return std::to_string(it->get<int64_t>());
    }
    return "";
}

std::string manifest_family(const json& record)
{
    return lowercase_ascii(json_string_value(record, "family"));
}

std::string manifest_subtype(const json& record)
{
    return lowercase_ascii(json_string_value(record, "subtype"));
}

bool manifest_record_allowed(const json& record,
                             const std::vector<std::string>& families,
                             const std::vector<std::string>& subtypes)
{
    const std::string family = manifest_family(record);
    const std::string subtype = manifest_subtype(record);
    if (!families.empty() && !contains_sorted_string(families, family)) {
        return false;
    }
    if (subtypes.empty()) {
        return true;
    }
    return contains_sorted_string(subtypes, subtype) ||
           contains_sorted_string(subtypes, family + "." + subtype);
}

std::string normalize_manifest_token_text(const std::string& text)
{
    std::string normalized;
    normalized.reserve(text.size());
    bool previous_space = false;
    for (unsigned char ch : text) {
        if (std::isalnum(ch)) {
            normalized.push_back(static_cast<char>(std::tolower(ch)));
            previous_space = false;
        } else if (!previous_space && !normalized.empty()) {
            normalized.push_back(' ');
            previous_space = true;
        }
    }
    while (!normalized.empty() && normalized.back() == ' ') {
        normalized.pop_back();
    }
    return normalized;
}

std::vector<std::string> token_ngrams(const std::string& normalized, uint32_t ngram_size)
{
    std::vector<std::string> grams;
    if (normalized.empty()) {
        return grams;
    }
    const uint32_t n = std::max<uint32_t>(1, ngram_size);
    if (normalized.size() <= n) {
        grams.push_back(normalized);
        return grams;
    }
    grams.reserve(normalized.size() - n + 1);
    for (size_t offset = 0; offset + n <= normalized.size(); ++offset) {
        grams.push_back(normalized.substr(offset, n));
    }
    std::sort(grams.begin(), grams.end());
    grams.erase(std::unique(grams.begin(), grams.end()), grams.end());
    return grams;
}

bool string_has_prefix(std::string_view value, std::string_view prefix)
{
    return value.rfind(prefix, 0) == 0;
}

bool string_has_suffix(std::string_view value, std::string_view suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

uint32_t sha256_rotate_right(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32u - bits));
}

uint64_t checked_binary_add(uint64_t lhs,
                            uint64_t rhs,
                            const std::string& context)
{
    if (rhs > std::numeric_limits<uint64_t>::max() - lhs) {
        fail_graph(context + " size arithmetic overflow");
    }
    return lhs + rhs;
}

uint64_t checked_binary_multiply(uint64_t lhs,
                                 uint64_t rhs,
                                 const std::string& context)
{
    if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        fail_graph(context + " size arithmetic overflow");
    }
    return lhs * rhs;
}

size_t checked_binary_element_count(uint64_t count,
                                    size_t element_size,
                                    const std::string& context)
{
    if (count > std::numeric_limits<size_t>::max() ||
        (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size)) {
        fail_graph(context + " exceeds addressable memory");
    }
    return static_cast<size_t>(count);
}

std::streamsize checked_binary_stream_bytes(size_t count,
                                            size_t element_size,
                                            const std::string& context)
{
    if (element_size != 0 && count > std::numeric_limits<size_t>::max() / element_size) {
        fail_graph(context + " byte count overflow");
    }
    const size_t bytes = count * element_size;
    if (bytes > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        fail_graph(context + " exceeds stream I/O limits");
    }
    return static_cast<std::streamsize>(bytes);
}

class Sha256Verifier {
public:
    void update(const uint8_t* data, size_t size)
    {
        total_bytes_ += size;
        while (size > 0) {
            const size_t copied = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, copied);
            block_size_ += copied;
            data += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish()
    {
        const uint64_t bit_count = total_bytes_ * 8u;
        std::array<uint8_t, 128> padding{};
        padding[0] = 0x80u;
        const size_t padding_size = block_size_ < 56 ? 56 - block_size_ : 120 - block_size_;
        update(padding.data(), padding_size);

        std::array<uint8_t, 8> encoded_length{};
        for (size_t idx = 0; idx < encoded_length.size(); ++idx) {
            encoded_length[encoded_length.size() - 1 - idx] =
                static_cast<uint8_t>((bit_count >> (idx * 8)) & 0xffu);
        }
        update(encoded_length.data(), encoded_length.size());

        std::array<uint8_t, 32> digest{};
        for (size_t word = 0; word < state_.size(); ++word) {
            digest[word * 4] = static_cast<uint8_t>(state_[word] >> 24);
            digest[word * 4 + 1] = static_cast<uint8_t>(state_[word] >> 16);
            digest[word * 4 + 2] = static_cast<uint8_t>(state_[word] >> 8);
            digest[word * 4 + 3] = static_cast<uint8_t>(state_[word]);
        }
        return digest;
    }

private:
    void transform(const uint8_t* block)
    {
        static constexpr std::array<uint32_t, 64> constants{
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
            0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
            0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
            0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
            0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
            0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
            0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
            0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
            0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        std::array<uint32_t, 64> words{};
        for (size_t idx = 0; idx < 16; ++idx) {
            const size_t offset = idx * 4;
            words[idx] = (static_cast<uint32_t>(block[offset]) << 24) |
                         (static_cast<uint32_t>(block[offset + 1]) << 16) |
                         (static_cast<uint32_t>(block[offset + 2]) << 8) |
                         static_cast<uint32_t>(block[offset + 3]);
        }
        for (size_t idx = 16; idx < words.size(); ++idx) {
            const uint32_t s0 = sha256_rotate_right(words[idx - 15], 7) ^
                                sha256_rotate_right(words[idx - 15], 18) ^
                                (words[idx - 15] >> 3);
            const uint32_t s1 = sha256_rotate_right(words[idx - 2], 17) ^
                                sha256_rotate_right(words[idx - 2], 19) ^
                                (words[idx - 2] >> 10);
            words[idx] = words[idx - 16] + s0 + words[idx - 7] + s1;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t idx = 0; idx < words.size(); ++idx) {
            const uint32_t sum1 = sha256_rotate_right(e, 6) ^
                                  sha256_rotate_right(e, 11) ^
                                  sha256_rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + sum1 + choice + constants[idx] + words[idx];
            const uint32_t sum0 = sha256_rotate_right(a, 2) ^
                                  sha256_rotate_right(a, 13) ^
                                  sha256_rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_{
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::array<uint8_t, 64> block_{};
    size_t block_size_ = 0;
    uint64_t total_bytes_ = 0;
};

std::array<uint8_t, 32> sha256_file_payload(const std::string& path,
                                            uint64_t offset,
                                            uint64_t payload_bytes)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open token operator index " + path + " for checksum validation");
    }
    if (offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        fail_graph("token sidecar payload offset exceeds stream limits in " + path);
    }
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!input) {
        fail_graph("unable to seek token operator payload in " + path);
    }
    Sha256Verifier verifier;
    std::array<uint8_t, 1024 * 1024> buffer{};
    uint64_t remaining = payload_bytes;
    while (remaining > 0) {
        const size_t requested = static_cast<size_t>(
            std::min<uint64_t>(remaining, static_cast<uint64_t>(buffer.size())));
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(requested));
        if (input.gcount() != static_cast<std::streamsize>(requested)) {
            fail_graph("truncated token operator payload in " + path);
        }
        verifier.update(buffer.data(), requested);
        remaining -= requested;
    }
    return verifier.finish();
}

constexpr char kTokenOperatorMagic[16] = {
    'F', 'A', 'N', 'N', 'S', 'O', 'P', 'L', 'E', 'X', 'I', 'D', 'X', '\0', '\0', '\0'};
constexpr uint32_t kTokenOperatorVersion = 1;
constexpr uint32_t kTokenOperatorEndianMarker = 0x01020304u;
constexpr uint32_t kTokenOperatorFlags = 0x7u;

struct TokenOperatorHeader {
    char magic[16];
    uint32_t version = 0;
    uint32_t header_bytes = 0;
    uint32_t endian = 0;
    uint32_t flags = 0;
    uint64_t token_count = 0;
    uint64_t prefix_count = 0;
    uint64_t suffix_count = 0;
    uint64_t cost_count = 0;
    unsigned char checksum_sha256[32];
};

static_assert(sizeof(TokenOperatorHeader) == 96, "TokenOperatorHeader layout mismatch");

struct TokenOperatorPayload {
    std::vector<uint32_t> prefix_order;
    std::vector<uint32_t> suffix_order;
    std::vector<uint32_t> posting_costs;
};

void validate_token_operator_permutation(const std::vector<uint32_t>& order,
                                         size_t token_count,
                                         const char* name,
                                         const std::string& path)
{
    std::vector<uint8_t> seen(token_count, 0);
    for (uint32_t token_id : order) {
        if (token_id >= token_count) {
            fail_graph(std::string("token operator ") + name +
                         " contains an out-of-range token id in " + path);
        }
        if (seen[token_id] != 0) {
            fail_graph(std::string("token operator ") + name +
                         " is not a token-id permutation in " + path);
        }
        seen[token_id] = 1;
    }
}

TokenOperatorPayload load_token_operator_payload(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open explicitly configured token operator index " + path);
    }
    TokenOperatorHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        fail_graph("invalid token operator index header in " + path);
    }
    if (std::memcmp(header.magic, kTokenOperatorMagic, sizeof(kTokenOperatorMagic)) != 0) {
        fail_graph("unsupported token operator magic in " + path);
    }
    if (header.version != kTokenOperatorVersion) {
        fail_graph("unsupported token operator version in " + path);
    }
    if (header.header_bytes != sizeof(TokenOperatorHeader)) {
        fail_graph("token operator header size mismatch in " + path);
    }
    if (header.endian != kTokenOperatorEndianMarker) {
        fail_graph("token operator endian marker mismatch in " + path);
    }
    if (header.flags != kTokenOperatorFlags) {
        fail_graph("token operator flags mismatch in " + path);
    }
    if (header.prefix_count != header.token_count ||
        header.suffix_count != header.token_count ||
        header.cost_count != header.token_count) {
        fail_graph("token operator counts must all equal token_count in " + path);
    }
    if (header.token_count > std::numeric_limits<uint32_t>::max() ||
        header.token_count > std::numeric_limits<size_t>::max()) {
        fail_graph("token operator token count exceeds the uint32 id space in " + path);
    }

    const uint64_t payload_bytes = checked_binary_multiply(
        header.token_count, 3u * sizeof(uint32_t), "token operator index " + path);
    const uint64_t expected_size = checked_binary_add(
        sizeof(TokenOperatorHeader), payload_bytes, "token operator index " + path);
    std::error_code ec;
    const uint64_t actual_size = std::filesystem::file_size(path, ec);
    if (ec) {
        fail_graph("failed to stat token operator index " + path);
    }
    if (actual_size != expected_size) {
        fail_graph("token operator index size mismatch in " + path +
                     ": actual=" + std::to_string(actual_size) +
                     " expected=" + std::to_string(expected_size));
    }

    const std::array<uint8_t, 32> actual_checksum =
        sha256_file_payload(path, sizeof(TokenOperatorHeader), payload_bytes);
    if (!std::equal(actual_checksum.begin(),
                    actual_checksum.end(),
                    std::begin(header.checksum_sha256))) {
        fail_graph("token operator payload checksum mismatch in " + path);
    }

    const size_t token_count = checked_binary_element_count(
        header.token_count, sizeof(uint32_t), "token operator arrays in " + path);
    TokenOperatorPayload payload;
    payload.prefix_order.resize(token_count);
    payload.suffix_order.resize(token_count);
    payload.posting_costs.resize(token_count);
    input.seekg(static_cast<std::streamoff>(sizeof(TokenOperatorHeader)), std::ios::beg);
    if (token_count > 0) {
        const std::streamsize array_bytes = checked_binary_stream_bytes(
            token_count, sizeof(uint32_t), "token operator arrays in " + path);
        input.read(reinterpret_cast<char*>(payload.prefix_order.data()), array_bytes);
        input.read(reinterpret_cast<char*>(payload.suffix_order.data()), array_bytes);
        input.read(reinterpret_cast<char*>(payload.posting_costs.data()), array_bytes);
    }
    if (!input) {
        fail_graph("failed to read token operator payload arrays from " + path);
    }
    validate_token_operator_permutation(payload.prefix_order, token_count, "prefix order", path);
    validate_token_operator_permutation(payload.suffix_order, token_count, "suffix order", path);
    return payload;
}

constexpr char kTokenNgramMagic[16] = {
    'F', 'A', 'N', 'N', 'S', 'N', 'G', 'R', 'A', 'M', 'I', 'D', 'X', '\0', '\0', '\0'};
constexpr uint32_t kTokenNgramVersion = 1;
constexpr uint32_t kTokenNgramEndianMarker = 0x01020304u;
constexpr uint64_t kTokenNgramChecksumKindSha256 = 1;
constexpr uint64_t kTokenNgramNormalizerVersion = 1;

struct TokenNgramHeader {
    char magic[16];
    uint32_t version = 0;
    uint32_t header_bytes = 0;
    uint32_t endian = 0;
    uint32_t flags = 0;
    uint64_t label_count = 0;
    uint64_t token_count = 0;
    uint64_t ngram_size = 0;
    uint64_t normalizer_version = 0;
    uint64_t token_text_bytes = 0;
    uint64_t ngram_count = 0;
    uint64_t ngram_text_bytes = 0;
    uint64_t posting_count = 0;
    uint64_t checksum_kind = 0;
    uint64_t reserved = 0;
    unsigned char checksum_sha256[32];
};

static_assert(sizeof(TokenNgramHeader) == 144, "TokenNgramHeader layout mismatch");

struct TokenNgramBinaryInfo {
    uint64_t label_count = 0;
    uint64_t token_count = 0;
    uint64_t ngram_size = 0;
    uint64_t normalizer_version = 0;
    uint64_t token_text_bytes = 0;
    uint64_t ngram_count = 0;
    uint64_t ngram_text_bytes = 0;
    uint64_t posting_count = 0;
    uint64_t token_ids_offset = 0;
    uint64_t token_ngram_counts_offset = 0;
    uint64_t token_text_offsets_offset = 0;
    uint64_t token_text_payload_offset = 0;
    uint64_t ngram_text_offsets_offset = 0;
    uint64_t ngram_offsets_offset = 0;
    uint64_t ngram_text_payload_offset = 0;
    uint64_t posting_ids_offset = 0;
};

TokenNgramBinaryInfo inspect_token_ngram_binary(const std::string& path,
                                                bool verify_payload_checksum)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open token ngram index " + path);
    }
    TokenNgramHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        fail_graph("invalid token ngram index header in " + path);
    }
    if (std::memcmp(header.magic, kTokenNgramMagic, sizeof(kTokenNgramMagic)) != 0) {
        fail_graph("unsupported token ngram magic in " + path);
    }
    if (header.version != kTokenNgramVersion) {
        fail_graph("unsupported token ngram version in " + path);
    }
    if (header.header_bytes != sizeof(TokenNgramHeader)) {
        fail_graph("token ngram header size mismatch in " + path);
    }
    if (header.endian != kTokenNgramEndianMarker) {
        fail_graph("token ngram endian marker mismatch in " + path);
    }
    if (header.checksum_kind != kTokenNgramChecksumKindSha256) {
        fail_graph("unsupported token ngram checksum kind in " + path);
    }
    if (header.normalizer_version != kTokenNgramNormalizerVersion) {
        fail_graph("unsupported token ngram normalizer version in " + path);
    }
    if (header.flags != 0 || header.reserved != 0) {
        fail_graph("unsupported token ngram flags or reserved fields in " + path);
    }
    if (header.label_count > std::numeric_limits<uint32_t>::max() ||
        header.token_count > header.label_count) {
        fail_graph("token ngram label/token counts exceed the uint32 label space in " + path);
    }
    if (header.ngram_size == 0 || header.ngram_size > std::numeric_limits<uint32_t>::max()) {
        fail_graph("token ngram size is outside the uint32 range in " + path);
    }

    const uint64_t token_offset_count = checked_binary_add(
        header.token_count, 1, "token ngram token offset count in " + path);
    const uint64_t ngram_offset_count = checked_binary_add(
        header.ngram_count, 1, "token ngram ngram offset count in " + path);
    checked_binary_element_count(
        header.token_count, sizeof(uint32_t), "token ngram token arrays in " + path);
    checked_binary_element_count(
        token_offset_count, sizeof(uint64_t), "token ngram token offsets in " + path);
    checked_binary_element_count(
        header.token_text_bytes, sizeof(char), "token ngram token text in " + path);
    checked_binary_element_count(
        ngram_offset_count, sizeof(uint64_t), "token ngram offsets in " + path);
    checked_binary_element_count(
        header.ngram_text_bytes, sizeof(char), "token ngram text in " + path);
    checked_binary_element_count(
        header.posting_count, sizeof(uint32_t), "token ngram postings in " + path);

    uint64_t offset = static_cast<uint64_t>(header.header_bytes);
    const uint64_t token_ids_offset = offset;
    offset = checked_binary_add(
        offset,
        checked_binary_multiply(header.token_count,
                                sizeof(uint32_t),
                                "token ngram token ids in " + path),
        "token ngram token ids in " + path);
    const uint64_t token_ngram_counts_offset = offset;
    offset = checked_binary_add(
        offset,
        checked_binary_multiply(header.token_count,
                                sizeof(uint32_t),
                                "token ngram counts in " + path),
        "token ngram counts in " + path);
    const uint64_t token_text_offsets_offset = offset;
    offset = checked_binary_add(
        offset,
        checked_binary_multiply(token_offset_count,
                                sizeof(uint64_t),
                                "token ngram token text offsets in " + path),
        "token ngram token text offsets in " + path);
    const uint64_t token_text_payload_offset = offset;
    offset = checked_binary_add(
        offset, header.token_text_bytes, "token ngram token text payload in " + path);
    const uint64_t ngram_text_offsets_offset = offset;
    offset = checked_binary_add(
        offset,
        checked_binary_multiply(ngram_offset_count,
                                sizeof(uint64_t),
                                "token ngram text offsets in " + path),
        "token ngram text offsets in " + path);
    const uint64_t ngram_offsets_offset = offset;
    offset = checked_binary_add(
        offset,
        checked_binary_multiply(ngram_offset_count,
                                sizeof(uint64_t),
                                "token ngram posting offsets in " + path),
        "token ngram posting offsets in " + path);
    const uint64_t ngram_text_payload_offset = offset;
    offset = checked_binary_add(
        offset, header.ngram_text_bytes, "token ngram text payload in " + path);
    const uint64_t posting_ids_offset = offset;
    offset = checked_binary_add(
        offset,
        checked_binary_multiply(header.posting_count,
                                sizeof(uint32_t),
                                "token ngram posting ids in " + path),
        "token ngram posting ids in " + path);

    std::error_code ec;
    const uint64_t actual_size = std::filesystem::file_size(path, ec);
    if (ec) {
        fail_graph("failed to stat token ngram index " + path);
    }
    if (actual_size != offset) {
        fail_graph("token ngram index size mismatch in " + path +
                     ": actual=" + std::to_string(actual_size) +
                     " expected=" + std::to_string(offset));
    }
    if (verify_payload_checksum) {
        const std::array<uint8_t, 32> actual_checksum = sha256_file_payload(
            path, header.header_bytes, offset - header.header_bytes);
        if (!std::equal(actual_checksum.begin(),
                        actual_checksum.end(),
                        std::begin(header.checksum_sha256))) {
            fail_graph("token ngram payload checksum mismatch in " + path);
        }
    }

    return TokenNgramBinaryInfo{
        header.label_count,
        header.token_count,
        header.ngram_size,
        header.normalizer_version,
        header.token_text_bytes,
        header.ngram_count,
        header.ngram_text_bytes,
        header.posting_count,
        token_ids_offset,
        token_ngram_counts_offset,
        token_text_offsets_offset,
        token_text_payload_offset,
        ngram_text_offsets_offset,
        ngram_offsets_offset,
        ngram_text_payload_offset,
        posting_ids_offset,
    };
}

std::optional<std::string> discover_token_ngram_index_path(const std::string& manifest_path)
{
    if (manifest_path.empty()) {
        return std::nullopt;
    }
    std::error_code ec;
    const std::filesystem::path candidate =
        std::filesystem::path(manifest_path).parent_path() / "token_ngram_index.bin";
    if (!std::filesystem::is_regular_file(candidate, ec) || ec) {
        return std::nullopt;
    }
    return candidate.string();
}

struct QueryManifestLabels {
    SparseLabelMatrix labels;
    std::vector<QueryLabelOp> ops;
    HybridQueryClauseTable clauses;
};

struct QueryManifestRowResult {
    std::vector<uint32_t> labels;
    std::vector<uint8_t> source_tiers;
    std::vector<uint32_t> relevance_ranks;
};

enum class HybridManifestAtomKind : uint8_t {
    Label = 0,
    Fuzzy = 1,
    Range = 2,
};

struct HybridManifestAtomWork {
    HybridManifestAtomKind kind = HybridManifestAtomKind::Label;
    uint32_t branch_id = 0;
    QueryManifestRowResult result;
    size_t shared_fuzzy_result = SIZE_MAX;
};

struct QueryManifestRowWork {
    json record;
    uint64_t line_number = 0;
    QueryLabelOp op = QueryLabelOp::And;
    bool fuzzy = false;
    bool hybrid = false;
    uint32_t branch_cap = 0;
    size_t shared_fuzzy_result = SIZE_MAX;
    std::vector<HybridManifestAtomWork> hybrid_atoms;
};

struct HybridFuzzyTarget {
    size_t selected_row = 0;
    size_t atom = 0;
};

class ManifestNgramIndex {
public:
    explicit ManifestNgramIndex(uint32_t ngram_size)
        : ngram_size_(std::max<uint32_t>(1, ngram_size))
    {
    }

    void add(uint32_t token_id, const std::string& text, uint32_t label_count)
    {
        if (token_id >= label_count) {
            return;
        }
        const std::string normalized = normalize_manifest_token_text(text);
        if (normalized.empty()) {
            return;
        }
        auto inserted = token_text_by_id_.emplace(token_id, normalized);
        if (!inserted.second) {
            return;
        }
        const std::vector<std::string> grams = token_ngrams(normalized, ngram_size_);
        token_ngram_count_by_id_[token_id] = grams.size();
        for (const std::string& gram : grams) {
            inverted_[gram].push_back(token_id);
        }
    }

    void finalize()
    {
        for (auto& entry : inverted_) {
            auto& ids = entry.second;
            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        }
    }

    std::vector<uint32_t> expand(const std::string& pattern,
                                 const std::string& match_mode,
                                 float min_score,
                                 uint32_t max_terms,
                                 std::vector<uint8_t>* out_source_tiers = nullptr,
                                 std::vector<uint32_t>* out_relevance_ranks = nullptr) const
    {
        if (out_source_tiers != nullptr) {
            out_source_tiers->clear();
        }
        if (out_relevance_ranks != nullptr) {
            out_relevance_ranks->clear();
        }
        const auto expand_start = LoadClock::now();
        if (max_terms == 0) {
            fail_graph("query_manifest_max_fuzzy_terms must be positive");
        }
        auto stage_start = LoadClock::now();
        const std::string normalized_pattern = normalize_manifest_token_text(pattern);
        if (normalized_pattern.empty()) {
            return {};
        }
        const std::vector<std::string> grams = token_ngrams(normalized_pattern, ngram_size_);
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_normalize_ngram_ns += elapsed_load_ns(stage_start);
            profile.fuzzy_patterns += 1;
            profile.pattern_chars += normalized_pattern.size();
            profile.grams_total += grams.size();
            profile.max_grams = std::max<uint64_t>(profile.max_grams, grams.size());
        }
        stage_start = LoadClock::now();
        std::unordered_map<uint32_t, uint32_t> overlap_by_id;
        uint64_t local_grams_found = 0;
        uint64_t local_grams_missing = 0;
        uint64_t local_postings = 0;
        for (const std::string& gram : grams) {
            const auto it = inverted_.find(gram);
            if (it == inverted_.end()) {
                ++local_grams_missing;
                continue;
            }
            ++local_grams_found;
            local_postings += it->second.size();
            for (uint32_t token_id : it->second) {
                ++overlap_by_id[token_id];
            }
        }
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_posting_lookup_ns += elapsed_load_ns(stage_start);
            profile.grams_found += local_grams_found;
            profile.grams_missing += local_grams_missing;
            profile.postings_scanned += local_postings;
            profile.overlap_candidates += overlap_by_id.size();
            profile.max_postings = std::max(profile.max_postings, local_postings);
            profile.max_overlap_candidates =
                std::max<uint64_t>(profile.max_overlap_candidates, overlap_by_id.size());
        }

        struct Candidate {
            uint32_t token_id = 0;
            float score = 0.0f;
            bool direct_match = false;
        };
        stage_start = LoadClock::now();
        std::vector<Candidate> candidates;
        candidates.reserve(overlap_by_id.size());
        const std::string normalized_mode = lowercase_ascii(match_mode);
        for (const auto& entry : overlap_by_id) {
            const uint32_t token_id = entry.first;
            const auto text_it = token_text_by_id_.find(token_id);
            if (text_it == token_text_by_id_.end()) {
                continue;
            }
            const std::string& token_text = text_it->second;
            bool direct = false;
            if (normalized_mode == "prefix") {
                direct = string_has_prefix(token_text, normalized_pattern);
            } else if (normalized_mode == "suffix") {
                direct = string_has_suffix(token_text, normalized_pattern);
            } else if (normalized_mode == "contains" || normalized_mode == "substring") {
                direct = token_text.find(normalized_pattern) != std::string::npos;
            }
            const auto count_it = token_ngram_count_by_id_.find(token_id);
            const size_t token_ngram_count = count_it == token_ngram_count_by_id_.end()
                ? 0
                : count_it->second;
            const size_t union_count = grams.size() + token_ngram_count - entry.second;
            const float score = union_count == 0
                ? 0.0f
                : static_cast<float>(entry.second) / static_cast<float>(union_count);
            if (!direct && score < min_score) {
                continue;
            }
            candidates.push_back({token_id, direct ? std::max(1.0f, score) : score, direct});
        }
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_score_filter_ns += elapsed_load_ns(stage_start);
            profile.scored_candidates += candidates.size();
            profile.max_scored_candidates =
                std::max<uint64_t>(profile.max_scored_candidates, candidates.size());
        }

        stage_start = LoadClock::now();
        std::sort(candidates.begin(),
                  candidates.end(),
                  [](const Candidate& lhs, const Candidate& rhs) {
                      if (lhs.direct_match != rhs.direct_match) {
                          return lhs.direct_match > rhs.direct_match;
                      }
                      if (lhs.score != rhs.score) {
                          return lhs.score > rhs.score;
                      }
                      return lhs.token_id < rhs.token_id;
                  });
        if (candidates.size() > max_terms) {
            candidates.resize(max_terms);
        }
        if (active_query_manifest_profile != nullptr) {
            active_query_manifest_profile->expand_candidate_sort_ns += elapsed_load_ns(stage_start);
        }
        stage_start = LoadClock::now();
        std::unordered_map<uint32_t, std::pair<uint8_t, uint32_t>> metadata_by_label;
        metadata_by_label.reserve(candidates.size());
        uint32_t fallback_rank = 1;
        std::vector<uint32_t> labels;
        labels.reserve(candidates.size());
        for (const Candidate& candidate : candidates) {
            labels.push_back(candidate.token_id);
            const QueryLabelSourceTier tier = candidate.direct_match
                ? QueryLabelSourceTier::Direct
                : QueryLabelSourceTier::Fallback;
            metadata_by_label.emplace(
                candidate.token_id,
                std::make_pair(static_cast<uint8_t>(tier),
                               candidate.direct_match ? 0U : fallback_rank++));
        }
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (out_source_tiers != nullptr) {
            out_source_tiers->reserve(labels.size());
        }
        if (out_relevance_ranks != nullptr) {
            out_relevance_ranks->reserve(labels.size());
        }
        for (uint32_t label : labels) {
            const auto metadata_it = metadata_by_label.find(label);
            if (metadata_it == metadata_by_label.end()) {
                fail_graph("manifest ngram occurrence metadata lost label alignment");
            }
            if (out_source_tiers != nullptr) {
                out_source_tiers->push_back(metadata_it->second.first);
            }
            if (out_relevance_ranks != nullptr) {
                out_relevance_ranks->push_back(metadata_it->second.second);
            }
        }
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_label_finalize_ns += elapsed_load_ns(stage_start);
            profile.emitted_labels += labels.size();
            profile.max_emitted_labels =
                std::max<uint64_t>(profile.max_emitted_labels, labels.size());
            const uint64_t expand_ns = elapsed_load_ns(expand_start);
            profile.max_expand_ns = std::max(profile.max_expand_ns, expand_ns);
        }
        return labels;
    }

    size_t token_count() const
    {
        return token_text_by_id_.size();
    }

private:
    uint32_t ngram_size_ = 3;
    std::unordered_map<uint32_t, std::string> token_text_by_id_;
    std::unordered_map<uint32_t, size_t> token_ngram_count_by_id_;
    std::unordered_map<std::string, std::vector<uint32_t>> inverted_;
};

class GlobalTokenNgramIndex {
public:
    explicit GlobalTokenNgramIndex(const std::string& path,
                                   bool verify_payload_checksum = false)
    {
        const TokenNgramBinaryInfo info =
            inspect_token_ngram_binary(path, verify_payload_checksum);
        ngram_size_ = static_cast<uint32_t>(info.ngram_size);
        label_count_ = static_cast<uint32_t>(info.label_count);

        std::ifstream input(path, std::ios::binary);
        if (!input) {
            fail_graph("unable to reopen token ngram index " + path);
        }

        std::vector<uint32_t> token_ids(checked_binary_element_count(
            info.token_count, sizeof(uint32_t), "token ngram token ids in " + path));
        token_ngram_counts_.resize(token_ids.size());
        token_text_offsets_.resize(checked_binary_element_count(
            checked_binary_add(info.token_count, 1, "token ngram token offsets in " + path),
            sizeof(uint64_t),
            "token ngram token offsets in " + path));
        token_text_payload_.resize(checked_binary_element_count(
            info.token_text_bytes, sizeof(char), "token ngram token text in " + path));
        const size_t ngram_offset_count = checked_binary_element_count(
            checked_binary_add(info.ngram_count, 1, "token ngram offsets in " + path),
            sizeof(uint64_t),
            "token ngram offsets in " + path);
        ngram_text_offsets_.resize(ngram_offset_count);
        ngram_offsets_.resize(ngram_offset_count);
        ngram_text_payload_.resize(checked_binary_element_count(
            info.ngram_text_bytes, sizeof(char), "token ngram text in " + path));
        posting_ids_.resize(checked_binary_element_count(
            info.posting_count, sizeof(uint32_t), "token ngram postings in " + path));

        input.seekg(static_cast<std::streamoff>(info.token_ids_offset), std::ios::beg);
        if (!input) {
            fail_graph("failed to seek token ids in " + path);
        }
        if (!token_ids.empty()) {
            input.read(reinterpret_cast<char*>(token_ids.data()),
                       checked_binary_stream_bytes(
                           token_ids.size(), sizeof(uint32_t), "token ngram token ids in " + path));
            input.read(reinterpret_cast<char*>(token_ngram_counts_.data()),
                       checked_binary_stream_bytes(token_ngram_counts_.size(),
                                                   sizeof(uint32_t),
                                                   "token ngram counts in " + path));
            input.read(reinterpret_cast<char*>(token_text_offsets_.data()),
                       checked_binary_stream_bytes(token_text_offsets_.size(),
                                                   sizeof(uint64_t),
                                                   "token ngram token offsets in " + path));
        }
        if (!token_text_payload_.empty()) {
            input.read(reinterpret_cast<char*>(token_text_payload_.data()),
                       static_cast<std::streamsize>(token_text_payload_.size()));
        }
        if (!ngram_text_offsets_.empty()) {
            input.read(reinterpret_cast<char*>(ngram_text_offsets_.data()),
                       checked_binary_stream_bytes(ngram_text_offsets_.size(),
                                                   sizeof(uint64_t),
                                                   "token ngram text offsets in " + path));
            input.read(reinterpret_cast<char*>(ngram_offsets_.data()),
                       checked_binary_stream_bytes(ngram_offsets_.size(),
                                                   sizeof(uint64_t),
                                                   "token ngram posting offsets in " + path));
        }
        if (!ngram_text_payload_.empty()) {
            input.read(reinterpret_cast<char*>(ngram_text_payload_.data()),
                       static_cast<std::streamsize>(ngram_text_payload_.size()));
        }
        if (!posting_ids_.empty()) {
            input.read(reinterpret_cast<char*>(posting_ids_.data()),
                       checked_binary_stream_bytes(posting_ids_.size(),
                                                   sizeof(uint32_t),
                                                   "token ngram postings in " + path));
        }
        if (!input) {
            fail_graph("failed to read token ngram payloads from " + path);
        }

        for (size_t idx = 0; idx < token_ids.size(); ++idx) {
            if (token_ids[idx] != idx) {
                fail_graph("token ngram index token ids must be contiguous in " + path);
            }
        }
    }

    std::vector<uint32_t> expand(const std::string& pattern,
                                 const std::string& match_mode,
                                 float min_score,
                                 uint32_t max_terms,
                                 std::vector<uint8_t>* out_source_tiers = nullptr,
                                 std::vector<uint32_t>* out_relevance_ranks = nullptr) const
    {
        if (out_source_tiers != nullptr) {
            out_source_tiers->clear();
        }
        if (out_relevance_ranks != nullptr) {
            out_relevance_ranks->clear();
        }
        const auto expand_start = LoadClock::now();
        if (max_terms == 0) {
            fail_graph("query_manifest_max_fuzzy_terms must be positive");
        }
        auto stage_start = LoadClock::now();
        const std::string normalized_pattern = normalize_manifest_token_text(pattern);
        if (normalized_pattern.empty()) {
            return {};
        }
        const std::vector<std::string> grams = token_ngrams(normalized_pattern, ngram_size_);
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_normalize_ngram_ns += elapsed_load_ns(stage_start);
            profile.fuzzy_patterns += 1;
            profile.pattern_chars += normalized_pattern.size();
            profile.grams_total += grams.size();
            profile.max_grams = std::max<uint64_t>(profile.max_grams, grams.size());
        }
        stage_start = LoadClock::now();
        std::unordered_map<uint32_t, uint32_t> overlap_by_id;
        uint64_t local_grams_found = 0;
        uint64_t local_grams_missing = 0;
        uint64_t local_postings = 0;
        for (const std::string& gram : grams) {
            const std::optional<size_t> gram_idx = find_ngram_index(gram);
            if (!gram_idx.has_value()) {
                ++local_grams_missing;
                continue;
            }
            ++local_grams_found;
            const uint64_t post_begin = ngram_offsets_[*gram_idx];
            const uint64_t post_end = ngram_offsets_[*gram_idx + 1];
            if (post_begin > post_end || post_end > posting_ids_.size()) {
                fail_graph("invalid token ngram posting offsets");
            }
            local_postings += post_end - post_begin;
            for (uint64_t idx = post_begin; idx < post_end; ++idx) {
                ++overlap_by_id[posting_ids_[idx]];
            }
        }
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_posting_lookup_ns += elapsed_load_ns(stage_start);
            profile.grams_found += local_grams_found;
            profile.grams_missing += local_grams_missing;
            profile.postings_scanned += local_postings;
            profile.overlap_candidates += overlap_by_id.size();
            profile.max_postings = std::max(profile.max_postings, local_postings);
            profile.max_overlap_candidates =
                std::max<uint64_t>(profile.max_overlap_candidates, overlap_by_id.size());
        }

        struct Candidate {
            uint32_t token_id = 0;
            float score = 0.0f;
            bool direct_match = false;
        };
        stage_start = LoadClock::now();
        std::vector<Candidate> candidates;
        candidates.reserve(overlap_by_id.size());
        const std::string normalized_mode = lowercase_ascii(match_mode);
        for (const auto& entry : overlap_by_id) {
            const uint32_t token_id = entry.first;
            if (token_id >= token_ngram_counts_.size()) {
                continue;
            }
            const std::string_view token_text = token_text_by_row(token_id);
            if (token_text.empty()) {
                continue;
            }
            bool direct = false;
            if (normalized_mode == "prefix") {
                direct = string_has_prefix(token_text, normalized_pattern);
            } else if (normalized_mode == "suffix") {
                direct = string_has_suffix(token_text, normalized_pattern);
            } else if (normalized_mode == "contains" || normalized_mode == "substring") {
                direct = token_text.find(normalized_pattern) != std::string_view::npos;
            }
            const size_t token_ngram_count = token_ngram_counts_[token_id];
            const size_t union_count = grams.size() + token_ngram_count - entry.second;
            const float score = union_count == 0
                ? 0.0f
                : static_cast<float>(entry.second) / static_cast<float>(union_count);
            if (!direct && score < min_score) {
                continue;
            }
            candidates.push_back({token_id, direct ? std::max(1.0f, score) : score, direct});
        }
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_score_filter_ns += elapsed_load_ns(stage_start);
            profile.scored_candidates += candidates.size();
            profile.max_scored_candidates =
                std::max<uint64_t>(profile.max_scored_candidates, candidates.size());
        }

        stage_start = LoadClock::now();
        std::sort(candidates.begin(),
                  candidates.end(),
                  [](const Candidate& lhs, const Candidate& rhs) {
                      if (lhs.direct_match != rhs.direct_match) {
                          return lhs.direct_match > rhs.direct_match;
                      }
                      if (lhs.score != rhs.score) {
                          return lhs.score > rhs.score;
                      }
                      return lhs.token_id < rhs.token_id;
                  });
        if (candidates.size() > max_terms) {
            candidates.resize(max_terms);
        }
        if (active_query_manifest_profile != nullptr) {
            active_query_manifest_profile->expand_candidate_sort_ns += elapsed_load_ns(stage_start);
        }
        stage_start = LoadClock::now();
        std::unordered_map<uint32_t, std::pair<uint8_t, uint32_t>> metadata_by_label;
        metadata_by_label.reserve(candidates.size());
        uint32_t fallback_rank = 1;
        std::vector<uint32_t> labels;
        labels.reserve(candidates.size());
        for (const Candidate& candidate : candidates) {
            if (candidate.token_id < label_count_) {
                labels.push_back(candidate.token_id);
                const QueryLabelSourceTier tier = candidate.direct_match
                    ? QueryLabelSourceTier::Direct
                    : QueryLabelSourceTier::Fallback;
                metadata_by_label.emplace(
                    candidate.token_id,
                    std::make_pair(static_cast<uint8_t>(tier),
                                   candidate.direct_match ? 0U : fallback_rank++));
            }
        }
        std::sort(labels.begin(), labels.end());
        labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
        if (out_source_tiers != nullptr) {
            out_source_tiers->reserve(labels.size());
        }
        if (out_relevance_ranks != nullptr) {
            out_relevance_ranks->reserve(labels.size());
        }
        for (uint32_t label : labels) {
            const auto metadata_it = metadata_by_label.find(label);
            if (metadata_it == metadata_by_label.end()) {
                fail_graph("global ngram occurrence metadata lost label alignment");
            }
            if (out_source_tiers != nullptr) {
                out_source_tiers->push_back(metadata_it->second.first);
            }
            if (out_relevance_ranks != nullptr) {
                out_relevance_ranks->push_back(metadata_it->second.second);
            }
        }
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.expand_label_finalize_ns += elapsed_load_ns(stage_start);
            profile.emitted_labels += labels.size();
            profile.max_emitted_labels =
                std::max<uint64_t>(profile.max_emitted_labels, labels.size());
            const uint64_t expand_ns = elapsed_load_ns(expand_start);
            profile.max_expand_ns = std::max(profile.max_expand_ns, expand_ns);
        }
        return labels;
    }

    size_t token_count() const
    {
        return token_ngram_counts_.size();
    }

    uint32_t label_count() const
    {
        return label_count_;
    }

    uint32_t ngram_size() const
    {
        return ngram_size_;
    }

    std::string_view token_text(size_t token_id) const
    {
        return token_text_by_row(token_id);
    }

    bool posting_list(const std::string& gram,
                      const uint32_t*& begin,
                      const uint32_t*& end) const
    {
        const std::optional<size_t> gram_idx = find_ngram_index(gram);
        if (!gram_idx.has_value()) {
            begin = nullptr;
            end = nullptr;
            return false;
        }
        const uint64_t post_begin = ngram_offsets_[*gram_idx];
        const uint64_t post_end = ngram_offsets_[*gram_idx + 1];
        if (post_begin > post_end || post_end > posting_ids_.size()) {
            fail_graph("invalid token ngram posting offsets");
        }
        begin = posting_ids_.data() + static_cast<size_t>(post_begin);
        end = posting_ids_.data() + static_cast<size_t>(post_end);
        return true;
    }

    void validate_operator_compatibility(const std::string& operator_path) const
    {
        if (token_count() > label_count_) {
            fail_graph("token ngram token count exceeds label count for operator index " +
                         operator_path);
        }
        if (token_text_offsets_.size() != token_count() + 1 ||
            token_text_offsets_.empty() || token_text_offsets_.front() != 0 ||
            token_text_offsets_.back() != token_text_payload_.size()) {
            fail_graph("token ngram token text offsets are incompatible with operator index " +
                         operator_path);
        }
        for (size_t idx = 1; idx < token_text_offsets_.size(); ++idx) {
            if (token_text_offsets_[idx - 1] > token_text_offsets_[idx]) {
                fail_graph("token ngram token text offsets are not monotone for operator index " +
                             operator_path);
            }
        }
        for (size_t token_id = 0; token_id < token_count(); ++token_id) {
            const std::string_view text = token_text_by_row(token_id);
            if (text.empty()) {
                fail_graph("token ngram contains an empty token text for operator index " +
                             operator_path);
            }
            for (unsigned char value : text) {
                const bool normalized = value == ' ' ||
                                        (value >= '0' && value <= '9') ||
                                        (value >= 'a' && value <= 'z');
                if (!normalized) {
                    fail_graph("token ngram contains non-normalized token text for operator index " +
                                 operator_path);
                }
            }
            if (token_ngram_counts_[token_id] == 0) {
                fail_graph("token ngram contains a token with zero ngrams for operator index " +
                             operator_path);
            }
        }
        if (ngram_text_offsets_.size() != ngram_offsets_.size() ||
            ngram_text_offsets_.empty() || ngram_text_offsets_.front() != 0 ||
            ngram_text_offsets_.back() != ngram_text_payload_.size() ||
            ngram_offsets_.front() != 0 || ngram_offsets_.back() != posting_ids_.size()) {
            fail_graph("token ngram sections are incompatible with operator index " + operator_path);
        }
        for (size_t idx = 1; idx < ngram_text_offsets_.size(); ++idx) {
            if (ngram_text_offsets_[idx - 1] > ngram_text_offsets_[idx] ||
                ngram_offsets_[idx - 1] > ngram_offsets_[idx]) {
                fail_graph("token ngram offsets are not monotone for operator index " + operator_path);
            }
        }
        std::vector<uint32_t> observed_ngram_counts(token_count(), 0);
        for (size_t gram_idx = 0; gram_idx + 1 < ngram_offsets_.size(); ++gram_idx) {
            const std::string_view gram_text = ngram_text_by_index(gram_idx);
            if (gram_text.empty()) {
                fail_graph("token ngram contains an empty ngram text for operator index " +
                             operator_path);
            }
            for (unsigned char value : gram_text) {
                const bool normalized = value == ' ' ||
                                        (value >= '0' && value <= '9') ||
                                        (value >= 'a' && value <= 'z');
                if (!normalized) {
                    fail_graph("token ngram contains non-normalized ngram text for operator index " +
                                 operator_path);
                }
            }
            if (gram_idx > 0 && ngram_text_by_index(gram_idx - 1) >= ngram_text_by_index(gram_idx)) {
                fail_graph("token ngram texts are not strictly sorted for operator index " +
                             operator_path);
            }
            const uint64_t begin = ngram_offsets_[gram_idx];
            const uint64_t end = ngram_offsets_[gram_idx + 1];
            if (begin == end) {
                fail_graph("token ngram contains an empty posting list for operator index " +
                             operator_path);
            }
            uint32_t previous_id = 0;
            bool have_previous = false;
            for (uint64_t idx = begin; idx < end; ++idx) {
                const uint32_t token_id = posting_ids_[idx];
                if (token_id >= token_count() || (have_previous && token_id <= previous_id)) {
                    fail_graph("token ngram postings are not sorted unique token ids for operator index " +
                                 operator_path);
                }
                if (observed_ngram_counts[token_id] == std::numeric_limits<uint32_t>::max()) {
                    fail_graph("token ngram posting frequency exceeds uint32 for operator index " +
                                 operator_path);
                }
                ++observed_ngram_counts[token_id];
                previous_id = token_id;
                have_previous = true;
            }
        }
        if (observed_ngram_counts != token_ngram_counts_) {
            fail_graph("token ngram counts do not match postings for operator index " +
                         operator_path);
        }
    }

private:
    std::string_view token_text_by_row(size_t row_idx) const
    {
        if (row_idx + 1 >= token_text_offsets_.size()) {
            return std::string_view();
        }
        const uint64_t begin = token_text_offsets_[row_idx];
        const uint64_t end = token_text_offsets_[row_idx + 1];
        if (begin > end || end > token_text_payload_.size()) {
            fail_graph("invalid token text offsets in token ngram index");
        }
        return std::string_view(token_text_payload_.data() + begin,
                                static_cast<size_t>(end - begin));
    }

    std::string_view ngram_text_by_index(size_t gram_idx) const
    {
        if (gram_idx + 1 >= ngram_text_offsets_.size()) {
            return std::string_view();
        }
        const uint64_t begin = ngram_text_offsets_[gram_idx];
        const uint64_t end = ngram_text_offsets_[gram_idx + 1];
        if (begin > end || end > ngram_text_payload_.size()) {
            fail_graph("invalid ngram text offsets in token ngram index");
        }
        return std::string_view(ngram_text_payload_.data() + begin,
                                static_cast<size_t>(end - begin));
    }

    std::optional<size_t> find_ngram_index(const std::string& gram) const
    {
        size_t left = 0;
        size_t right = ngram_offsets_.empty() ? 0 : ngram_offsets_.size() - 1;
        while (left < right) {
            const size_t mid = left + ((right - left) / 2);
            const int cmp = ngram_text_by_index(mid).compare(gram);
            if (cmp < 0) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        if (left >= ngram_offsets_.size() - 1) {
            return std::nullopt;
        }
        if (ngram_text_by_index(left) != gram) {
            return std::nullopt;
        }
        return left;
    }

    uint32_t ngram_size_ = 3;
    uint32_t label_count_ = 0;
    std::vector<uint32_t> token_ngram_counts_;
    std::vector<uint64_t> token_text_offsets_;
    std::string token_text_payload_;
    std::vector<uint64_t> ngram_text_offsets_;
    std::vector<uint64_t> ngram_offsets_;
    std::string ngram_text_payload_;
    std::vector<uint32_t> posting_ids_;
};

int compare_reversed_text(std::string_view lhs, std::string_view rhs)
{
    const size_t common = std::min(lhs.size(), rhs.size());
    for (size_t idx = 0; idx < common; ++idx) {
        const unsigned char lhs_ch = static_cast<unsigned char>(lhs[lhs.size() - 1 - idx]);
        const unsigned char rhs_ch = static_cast<unsigned char>(rhs[rhs.size() - 1 - idx]);
        if (lhs_ch < rhs_ch) {
            return -1;
        }
        if (lhs_ch > rhs_ch) {
            return 1;
        }
    }
    if (lhs.size() < rhs.size()) {
        return -1;
    }
    if (lhs.size() > rhs.size()) {
        return 1;
    }
    return 0;
}

struct OperatorDirectLookup {
    std::vector<uint32_t> matches;
    uint64_t candidates = 0;
};

struct OperatorExpandStats {
    bool used_fallback = false;
    uint64_t direct_candidates = 0;
    uint64_t direct_matches = 0;
};

class GlobalTokenOperatorIndex {
public:
    GlobalTokenOperatorIndex(const std::string& path,
                             const GlobalTokenNgramIndex& ngram_index,
                             bool parity_check,
                             QueryManifestOperatorFillMode fill_mode)
        : ngram_index_(ngram_index), parity_check_(parity_check), fill_mode_(fill_mode)
    {
        validate_query_manifest_operator_rewrite_options(
            query_manifest_operator_fill_mode_name(fill_mode_), parity_check_);
        TokenOperatorPayload payload = load_token_operator_payload(path);
        if (payload.posting_costs.size() != ngram_index_.token_count()) {
            fail_graph("token operator token_count=" +
                         std::to_string(payload.posting_costs.size()) +
                         " does not match GlobalTokenNgramIndex token_count=" +
                         std::to_string(ngram_index_.token_count()) + " in " + path);
        }
        ngram_index_.validate_operator_compatibility(path);
        prefix_order_ = std::move(payload.prefix_order);
        suffix_order_ = std::move(payload.suffix_order);
        posting_costs_ = std::move(payload.posting_costs);
        validate_lexical_order(prefix_order_, false, "prefix order", path);
        validate_lexical_order(suffix_order_, true, "suffix order", path);
    }

    std::vector<uint32_t> expand(const std::string& pattern,
                                 const std::string& match_mode,
                                 float min_score,
                                 uint32_t max_terms,
                                 std::vector<uint8_t>* out_source_tiers = nullptr,
                                 std::vector<uint32_t>* out_relevance_ranks = nullptr) const
    {
        return expand_with_stats(pattern,
                                 match_mode,
                                 min_score,
                                 max_terms,
                                 nullptr,
                                 out_source_tiers,
                                 out_relevance_ranks);
    }

    std::vector<uint32_t> expand_with_stats(const std::string& pattern,
                                            const std::string& match_mode,
                                            float min_score,
                                            uint32_t max_terms,
                                            OperatorExpandStats* out_stats,
                                            std::vector<uint8_t>* out_source_tiers = nullptr,
                                            std::vector<uint32_t>* out_relevance_ranks = nullptr) const
    {
        if (out_source_tiers != nullptr) {
            out_source_tiers->clear();
        }
        if (out_relevance_ranks != nullptr) {
            out_relevance_ranks->clear();
        }
        if (max_terms == 0) {
            fail_graph("query_manifest_max_fuzzy_terms must be positive");
        }
        const auto lookup_start = LoadClock::now();
        const std::string normalized_pattern = normalize_manifest_token_text(pattern);
        const std::string normalized_mode = lowercase_ascii(match_mode);
        OperatorDirectLookup direct;
        bool direct_only_compatible = !normalized_pattern.empty();
        if (normalized_mode == "prefix") {
            direct_only_compatible = direct_only_compatible &&
                                     (fill_mode_ == QueryManifestOperatorFillMode::LiteralOnly ||
                                      normalized_pattern.size() >= ngram_index_.ngram_size());
            if (direct_only_compatible) {
                direct = lexicon_matches(normalized_pattern, false);
            }
        } else if (normalized_mode == "suffix") {
            direct_only_compatible = direct_only_compatible &&
                                     (fill_mode_ == QueryManifestOperatorFillMode::LiteralOnly ||
                                      normalized_pattern.size() >= ngram_index_.ngram_size());
            if (direct_only_compatible) {
                direct = lexicon_matches(normalized_pattern, true);
            }
        } else if (normalized_mode == "contains" || normalized_mode == "substring") {
            if (fill_mode_ == QueryManifestOperatorFillMode::DirectThenFallback) {
                direct_only_compatible = direct_only_compatible &&
                                         normalized_pattern.size() >= ngram_index_.ngram_size();
            }
            if (direct_only_compatible) {
                direct = fill_mode_ == QueryManifestOperatorFillMode::LiteralOnly &&
                                 normalized_pattern.size() < ngram_index_.ngram_size()
                    ? contains_matches_by_vocabulary_scan(normalized_pattern)
                    : contains_matches(normalized_pattern);
            }
        } else {
            if (fill_mode_ == QueryManifestOperatorFillMode::LiteralOnly) {
                fail_graph(
                    "query_manifest_operator_fill_mode=literal_only supports only "
                    "prefix, suffix, and contains match modes");
            }
            direct_only_compatible = false;
        }
        const uint64_t lookup_ns = elapsed_load_ns(lookup_start);
        if (active_query_manifest_profile != nullptr) {
            QueryManifestLoadProfile& profile = *active_query_manifest_profile;
            profile.operator_lookup_ns += lookup_ns;
            profile.operator_lookups += 1;
            profile.operator_direct_candidates += direct.candidates;
            profile.operator_direct_matches += direct.matches.size();
        }

        std::sort(direct.matches.begin(), direct.matches.end());
        direct.matches.erase(std::unique(direct.matches.begin(), direct.matches.end()),
                             direct.matches.end());

        if (fill_mode_ == QueryManifestOperatorFillMode::LiteralOnly) {
            const uint64_t direct_match_count = direct.matches.size();
            if (out_source_tiers != nullptr) {
                out_source_tiers->assign(
                    direct.matches.size(), static_cast<uint8_t>(QueryLabelSourceTier::Direct));
            }
            if (out_relevance_ranks != nullptr) {
                out_relevance_ranks->reserve(direct.matches.size());
                for (uint32_t token_id : direct.matches) {
                    const size_t token_size = ngram_index_.token_text(token_id).size();
                    const size_t extra_chars = token_size >= normalized_pattern.size()
                        ? token_size - normalized_pattern.size()
                        : std::numeric_limits<uint32_t>::max();
                    out_relevance_ranks->push_back(
                        static_cast<uint32_t>(std::min<size_t>(
                            extra_chars, std::numeric_limits<uint32_t>::max())));
                }
            }
            if (active_query_manifest_profile != nullptr) {
                QueryManifestLoadProfile& profile = *active_query_manifest_profile;
                profile.operator_direct_only_hits += 1;
                profile.fuzzy_patterns += 1;
                profile.pattern_chars += normalized_pattern.size();
                profile.emitted_labels += direct.matches.size();
                profile.max_emitted_labels =
                    std::max<uint64_t>(profile.max_emitted_labels, direct.matches.size());
            }
            if (out_stats != nullptr) {
                out_stats->used_fallback = false;
                out_stats->direct_candidates = direct.candidates;
                out_stats->direct_matches = direct_match_count;
            }
            return direct.matches;
        }

        if (fill_mode_ == QueryManifestOperatorFillMode::DirectThenFallback) {
            return expand_direct_then_fallback(pattern,
                                               match_mode,
                                               min_score,
                                               max_terms,
                                               direct,
                                               direct_only_compatible,
                                               out_stats,
                                               out_source_tiers,
                                               out_relevance_ranks);
        }

        if (direct_only_compatible && direct.matches.size() >= max_terms) {
            const uint64_t direct_match_count = direct.matches.size();
            direct.matches.resize(max_terms);
            if (out_source_tiers != nullptr) {
                out_source_tiers->assign(
                    direct.matches.size(), static_cast<uint8_t>(QueryLabelSourceTier::Direct));
            }
            if (out_relevance_ranks != nullptr) {
                out_relevance_ranks->assign(direct.matches.size(), 0);
            }
            if (active_query_manifest_profile != nullptr) {
                QueryManifestLoadProfile& profile = *active_query_manifest_profile;
                profile.operator_direct_only_hits += 1;
                profile.fuzzy_patterns += 1;
                profile.pattern_chars += normalized_pattern.size();
                profile.emitted_labels += direct.matches.size();
                profile.max_emitted_labels =
                    std::max<uint64_t>(profile.max_emitted_labels, direct.matches.size());
            }
            if (out_stats != nullptr) {
                out_stats->used_fallback = false;
                out_stats->direct_candidates = direct.candidates;
                out_stats->direct_matches = direct_match_count;
            }
            return direct.matches;
        }

        QueryManifestLoadProfile* profile = active_query_manifest_profile;
        const uint64_t grams_before = profile == nullptr ? 0 : profile->grams_total;
        const uint64_t postings_before = profile == nullptr ? 0 : profile->postings_scanned;
        const uint64_t candidates_before = profile == nullptr ? 0 : profile->overlap_candidates;
        std::vector<uint32_t> labels =
            ngram_index_.expand(pattern,
                                match_mode,
                                min_score,
                                max_terms,
                                out_source_tiers,
                                out_relevance_ranks);
        if (profile != nullptr) {
            profile->operator_fallbacks += 1;
            profile->operator_baseline_grams += profile->grams_total - grams_before;
            profile->operator_baseline_postings += profile->postings_scanned - postings_before;
            profile->operator_baseline_candidates +=
                profile->overlap_candidates - candidates_before;
        }
        if (out_stats != nullptr) {
            out_stats->used_fallback = true;
            out_stats->direct_candidates = direct.candidates;
            out_stats->direct_matches = direct.matches.size();
        }
        return labels;
    }

    const GlobalTokenNgramIndex& baseline_index() const
    {
        return ngram_index_;
    }

    bool parity_check_enabled() const
    {
        return parity_check_;
    }

    size_t token_count() const
    {
        return posting_costs_.size();
    }

    uint32_t posting_cardinality(uint32_t label) const
    {
        if (label >= posting_costs_.size()) {
            fail_graph("token operator posting cardinality label is out of range");
        }
        return posting_costs_[label];
    }

    QueryManifestOperatorFillMode fill_mode() const
    {
        return fill_mode_;
    }

private:
    std::vector<uint32_t> expand_direct_then_fallback(
        const std::string& pattern,
        const std::string& match_mode,
        float min_score,
        uint32_t max_terms,
        const OperatorDirectLookup& direct,
        bool direct_compatible,
        OperatorExpandStats* out_stats,
        std::vector<uint8_t>* out_source_tiers,
        std::vector<uint32_t>* out_relevance_ranks) const
    {
        struct SelectedOccurrence {
            uint32_t label = 0;
            QueryLabelSourceTier source_tier = QueryLabelSourceTier::Fallback;
            uint32_t relevance_rank = 0;
        };

        std::vector<SelectedOccurrence> selected;
        selected.reserve(max_terms);
        std::unordered_set<uint32_t> selected_labels;
        selected_labels.reserve(max_terms);
        if (direct_compatible) {
            for (uint32_t label : direct.matches) {
                if (selected.size() >= max_terms) {
                    break;
                }
                selected.push_back({label, QueryLabelSourceTier::Direct, 0});
                selected_labels.insert(label);
            }
        }

        const bool used_fallback = selected.size() < max_terms;
        if (used_fallback) {
            QueryManifestLoadProfile* profile = active_query_manifest_profile;
            const uint64_t grams_before = profile == nullptr ? 0 : profile->grams_total;
            const uint64_t postings_before = profile == nullptr ? 0 : profile->postings_scanned;
            const uint64_t candidates_before = profile == nullptr ? 0 : profile->overlap_candidates;
            std::vector<uint8_t> baseline_tiers;
            std::vector<uint32_t> baseline_ranks;
            const std::vector<uint32_t> baseline_labels =
                ngram_index_.expand(pattern,
                                    match_mode,
                                    min_score,
                                    max_terms,
                                    &baseline_tiers,
                                    &baseline_ranks);
            if (baseline_tiers.size() != baseline_labels.size() ||
                baseline_ranks.size() != baseline_labels.size()) {
                fail_graph("baseline fuzzy occurrence metadata is not label-aligned");
            }
            if (profile != nullptr) {
                profile->operator_fallbacks += 1;
                profile->operator_baseline_grams += profile->grams_total - grams_before;
                profile->operator_baseline_postings += profile->postings_scanned - postings_before;
                profile->operator_baseline_candidates +=
                    profile->overlap_candidates - candidates_before;
            }

            std::vector<size_t> baseline_order(baseline_labels.size());
            std::iota(baseline_order.begin(), baseline_order.end(), 0);
            std::sort(baseline_order.begin(), baseline_order.end(), [&](size_t left, size_t right) {
                if (baseline_ranks[left] != baseline_ranks[right]) {
                    return baseline_ranks[left] < baseline_ranks[right];
                }
                return baseline_labels[left] < baseline_labels[right];
            });
            for (size_t idx : baseline_order) {
                if (selected.size() >= max_terms) {
                    break;
                }
                if (!selected_labels.insert(baseline_labels[idx]).second) {
                    continue;
                }
                selected.push_back({baseline_labels[idx],
                                    QueryLabelSourceTier::Fallback,
                                    baseline_ranks[idx]});
            }
        } else if (active_query_manifest_profile != nullptr) {
            active_query_manifest_profile->operator_direct_only_hits += 1;
            active_query_manifest_profile->fuzzy_patterns += 1;
            active_query_manifest_profile->pattern_chars +=
                normalize_manifest_token_text(pattern).size();
            active_query_manifest_profile->emitted_labels += selected.size();
            active_query_manifest_profile->max_emitted_labels = std::max<uint64_t>(
                active_query_manifest_profile->max_emitted_labels, selected.size());
        }

        std::sort(selected.begin(), selected.end(), [](const auto& left, const auto& right) {
            return left.label < right.label;
        });
        std::vector<uint32_t> labels;
        labels.reserve(selected.size());
        if (out_source_tiers != nullptr) {
            out_source_tiers->reserve(selected.size());
        }
        if (out_relevance_ranks != nullptr) {
            out_relevance_ranks->reserve(selected.size());
        }
        for (const SelectedOccurrence& occurrence : selected) {
            labels.push_back(occurrence.label);
            if (out_source_tiers != nullptr) {
                out_source_tiers->push_back(static_cast<uint8_t>(occurrence.source_tier));
            }
            if (out_relevance_ranks != nullptr) {
                out_relevance_ranks->push_back(occurrence.relevance_rank);
            }
        }
        if (out_stats != nullptr) {
            out_stats->used_fallback = used_fallback;
            out_stats->direct_candidates = direct.candidates;
            out_stats->direct_matches = direct.matches.size();
        }
        return labels;
    }

    int compare_order_key(uint32_t token_id, std::string_view key, bool reverse) const
    {
        const std::string_view token_text = ngram_index_.token_text(token_id);
        return reverse ? compare_reversed_text(token_text, key) : token_text.compare(key);
    }

    size_t lower_bound_order(const std::vector<uint32_t>& order,
                             std::string_view key,
                             bool reverse) const
    {
        size_t left = 0;
        size_t right = order.size();
        while (left < right) {
            const size_t middle = left + ((right - left) / 2);
            if (compare_order_key(order[middle], key, reverse) < 0) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }
        return left;
    }

    OperatorDirectLookup lexicon_matches(const std::string& pattern, bool reverse) const
    {
        const std::vector<uint32_t>& order = reverse ? suffix_order_ : prefix_order_;
        const size_t begin = lower_bound_order(order, pattern, reverse);
        OperatorDirectLookup result;
        for (size_t idx = begin; idx < order.size(); ++idx) {
            const uint32_t token_id = order[idx];
            const std::string_view token_text = ngram_index_.token_text(token_id);
            const bool matches = reverse ? string_has_suffix(token_text, pattern)
                                         : string_has_prefix(token_text, pattern);
            if (!matches) {
                break;
            }
            result.matches.push_back(token_id);
        }
        result.candidates = result.matches.size();
        return result;
    }

    OperatorDirectLookup contains_matches(const std::string& pattern) const
    {
        struct PostingSpan {
            const uint32_t* begin = nullptr;
            const uint32_t* end = nullptr;
        };
        std::vector<PostingSpan> spans;
        for (const std::string& gram : token_ngrams(pattern, ngram_index_.ngram_size())) {
            PostingSpan span;
            if (!ngram_index_.posting_list(gram, span.begin, span.end)) {
                return {};
            }
            if (span.begin == span.end) {
                return {};
            }
            spans.push_back(span);
        }
        if (spans.empty()) {
            return {};
        }
        std::sort(spans.begin(), spans.end(), [](const PostingSpan& lhs, const PostingSpan& rhs) {
            return (lhs.end - lhs.begin) < (rhs.end - rhs.begin);
        });
        std::vector<uint32_t> candidates(spans.front().begin, spans.front().end);
        for (size_t idx = 1; idx < spans.size() && !candidates.empty(); ++idx) {
            std::vector<uint32_t> intersection;
            intersection.reserve(std::min<size_t>(
                candidates.size(), static_cast<size_t>(spans[idx].end - spans[idx].begin)));
            std::set_intersection(candidates.begin(),
                                  candidates.end(),
                                  spans[idx].begin,
                                  spans[idx].end,
                                  std::back_inserter(intersection));
            candidates.swap(intersection);
        }
        OperatorDirectLookup result;
        result.candidates = candidates.size();
        for (uint32_t token_id : candidates) {
            if (ngram_index_.token_text(token_id).find(pattern) != std::string_view::npos) {
                result.matches.push_back(token_id);
            }
        }
        return result;
    }

    OperatorDirectLookup contains_matches_by_vocabulary_scan(
        const std::string& pattern) const
    {
        OperatorDirectLookup result;
        result.candidates = ngram_index_.token_count();
        for (uint32_t token_id = 0; token_id < ngram_index_.token_count(); ++token_id) {
            if (ngram_index_.token_text(token_id).find(pattern) != std::string_view::npos) {
                result.matches.push_back(token_id);
            }
        }
        return result;
    }

    void validate_lexical_order(const std::vector<uint32_t>& order,
                                bool reverse,
                                const char* name,
                                const std::string& path) const
    {
        for (size_t idx = 1; idx < order.size(); ++idx) {
            const uint32_t previous_id = order[idx - 1];
            const uint32_t current_id = order[idx];
            const std::string_view previous_text = ngram_index_.token_text(previous_id);
            const std::string_view current_text = ngram_index_.token_text(current_id);
            const int compared = reverse ? compare_reversed_text(previous_text, current_text)
                                         : previous_text.compare(current_text);
            if (compared > 0 || (compared == 0 && previous_id >= current_id)) {
                fail_graph(std::string("token operator ") + name +
                             " is not sorted for the GlobalTokenNgramIndex vocabulary in " + path);
            }
        }
    }

    const GlobalTokenNgramIndex& ngram_index_;
    bool parity_check_ = false;
    QueryManifestOperatorFillMode fill_mode_ =
        QueryManifestOperatorFillMode::LegacyFullFallback;
    std::vector<uint32_t> prefix_order_;
    std::vector<uint32_t> suffix_order_;
    std::vector<uint32_t> posting_costs_;
};

struct QueryManifestOperatorRuntimeSelection {
    std::string path;
    bool parity_check = false;
    QueryManifestOperatorFillMode fill_mode =
        QueryManifestOperatorFillMode::LegacyFullFallback;
};

std::mutex& query_manifest_operator_selection_mutex()
{
    static std::mutex mutex;
    return mutex;
}

QueryManifestOperatorRuntimeSelection& query_manifest_operator_selection()
{
    static QueryManifestOperatorRuntimeSelection selection;
    return selection;
}

void configure_query_manifest_operator_rewrite(const std::string& operator_index_path,
                                               bool parity_check,
                                               const std::string& fill_mode)
{
    validate_query_manifest_operator_rewrite_options(fill_mode, parity_check);
    const QueryManifestOperatorFillMode parsed_fill_mode =
        *parse_query_manifest_operator_fill_mode(fill_mode);
    std::lock_guard<std::mutex> lock(query_manifest_operator_selection_mutex());
    query_manifest_operator_selection().path = operator_index_path;
    query_manifest_operator_selection().parity_check = parity_check;
    query_manifest_operator_selection().fill_mode = parsed_fill_mode;
}

QueryManifestOperatorRuntimeSelection snapshot_query_manifest_operator_selection()
{
    std::lock_guard<std::mutex> lock(query_manifest_operator_selection_mutex());
    return query_manifest_operator_selection();
}

FuzzyOperatorRewriteTestResult evaluate_fuzzy_operator_rewrite_for_test(
    const std::string& token_ngram_index_path,
    const std::string& operator_index_path,
    const std::string& pattern,
    const std::string& match_mode,
    float min_score,
    uint32_t max_terms,
    const std::string& fill_mode)
{
    validate_query_manifest_operator_rewrite_options(fill_mode, false);
    GlobalTokenNgramIndex ngram_index(token_ngram_index_path, true);
    GlobalTokenOperatorIndex operator_index(
        operator_index_path,
        ngram_index,
        false,
        *parse_query_manifest_operator_fill_mode(fill_mode));
    OperatorExpandStats stats;
    FuzzyOperatorRewriteTestResult result;
    result.operator_labels = operator_index.expand_with_stats(
        pattern,
        match_mode,
        min_score,
        max_terms,
        &stats,
        &result.operator_source_tiers,
        &result.operator_relevance_ranks);
    result.baseline_labels = ngram_index.expand(pattern, match_mode, min_score, max_terms);
    result.used_fallback = stats.used_fallback;
    result.direct_candidates = stats.direct_candidates;
    result.direct_matches = stats.direct_matches;
    return result;
}

void exercise_fuzzy_operator_rewrite_concurrently_for_test(
    const std::string& token_ngram_index_path,
    const std::string& operator_index_path,
    const std::string& pattern,
    const std::string& match_mode,
    float min_score,
    uint32_t max_terms,
    uint32_t thread_count,
    uint32_t repetitions_per_thread,
    const std::string& fill_mode)
{
    if (thread_count == 0 || repetitions_per_thread == 0) {
        fail_graph("operator rewrite concurrency test requires positive thread and repetition counts");
    }
    validate_query_manifest_operator_rewrite_options(fill_mode, false);
    GlobalTokenNgramIndex ngram_index(token_ngram_index_path, true);
    GlobalTokenOperatorIndex operator_index(
        operator_index_path,
        ngram_index,
        false,
        *parse_query_manifest_operator_fill_mode(fill_mode));
    const std::vector<uint32_t> expected = operator_index.expand(
        pattern, match_mode, min_score, max_terms);

    std::vector<std::exception_ptr> errors(thread_count);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (uint32_t worker_id = 0; worker_id < thread_count; ++worker_id) {
        workers.emplace_back([&, worker_id] {
            try {
                QueryManifestLoadProfile worker_profile;
                ScopedQueryManifestProfile scoped_profile(&worker_profile);
                for (uint32_t repetition = 0;
                     repetition < repetitions_per_thread;
                     ++repetition) {
                    if (operator_index.expand(pattern, match_mode, min_score, max_terms) != expected) {
                        fail_graph("operator rewrite concurrency test produced a parity mismatch");
                    }
                }
            } catch (...) {
                errors[worker_id] = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (const std::exception_ptr& error : errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
}

uint32_t predicate_token_id(const json& predicate, uint32_t label_count, uint64_t line_number)
{
    const auto it = predicate.find("token_id");
    if (it == predicate.end() || it->is_null()) {
        fail_graph("query manifest line " + std::to_string(line_number) +
                     " predicate is missing token_id");
    }
    const uint64_t token_id = it->get<uint64_t>();
    if (token_id >= label_count) {
        fail_graph("query manifest line " + std::to_string(line_number) +
                     " token_id out of range");
    }
    return static_cast<uint32_t>(token_id);
}

std::vector<uint32_t> extract_manifest_token_ids(const json& record,
                                                 uint32_t label_count,
                                                 uint64_t line_number)
{
    const auto predicates_it = record.find("predicates");
    if (predicates_it == record.end() || !predicates_it->is_array()) {
        fail_graph("query manifest line " + std::to_string(line_number) +
                     " has no predicates array");
    }
    std::vector<uint32_t> labels;
    for (const json& predicate : *predicates_it) {
        labels.push_back(predicate_token_id(predicate, label_count, line_number));
    }
    std::sort(labels.begin(), labels.end());
    labels.erase(std::unique(labels.begin(), labels.end()), labels.end());
    if (labels.empty()) {
        fail_graph("query manifest line " + std::to_string(line_number) +
                     " has no token labels");
    }
    return labels;
}

void add_manifest_token_texts(const json& record,
                              ManifestNgramIndex& ngram_index,
                              uint32_t label_count)
{
    const auto predicates_it = record.find("predicates");
    if (predicates_it == record.end() || !predicates_it->is_array()) {
        return;
    }
    for (const json& predicate : *predicates_it) {
        const auto children_it = predicate.find("children");
        if (children_it != predicate.end() && children_it->is_array()) {
            for (const json& child : *children_it) {
                if (child.is_object()) {
                    add_manifest_token_texts(child, ngram_index, label_count);
                }
            }
        }
        const auto token_it = predicate.find("token_id");
        if (token_it != predicate.end() && !token_it->is_null()) {
            const uint64_t token_id = token_it->get<uint64_t>();
            const std::string token_text = json_string_value(predicate, "token_text");
            if (!token_text.empty() && token_id < label_count) {
                ngram_index.add(static_cast<uint32_t>(token_id), token_text, label_count);
            }
        }
        const auto source_token_it = predicate.find("source_token_id");
        if (source_token_it != predicate.end() && !source_token_it->is_null()) {
            const uint64_t token_id = source_token_it->get<uint64_t>();
            const std::string token_text = json_string_value(predicate, "source_token_text");
            if (!token_text.empty() && token_id < label_count) {
                ngram_index.add(static_cast<uint32_t>(token_id), token_text, label_count);
            }
        }
    }
}

std::string fuzzy_pattern_from_predicate(const json& predicate)
{
    const char* keys[] = {"pattern", "query", "value", "token_text", "source_token_text"};
    for (const char* key : keys) {
        const std::string value = json_string_value(predicate, key);
        if (!value.empty()) {
            return value;
        }
    }
    return "";
}

json lower_fuzzy_work_record(const json& record, uint64_t line)
{
    const auto it = record.find("predicates");
    if (it == record.end() || !it->is_array())
        fail_graph("query manifest line " + std::to_string(line) + " fuzzy query has no predicates array");
    json result = {{"predicates", json::array()}};
    for (const auto& predicate : *it) {
        std::string mode = json_string_value(predicate, "match_mode");
        if (mode.empty()) mode = json_string_value(predicate, "fuzzy_match_mode");
        result["predicates"].push_back({{"match_mode", mode}, {"pattern", fuzzy_pattern_from_predicate(predicate)}});
    }
    return result;
}

// Index identity and semantic options are constant for this invocation. The
// dictionary lives only until the current incoming batch has been prepared.
std::string fuzzy_batch_atom_key(const QueryManifestRowWork& work)
{
    std::string key;
    for (const auto& predicate : work.record.at("predicates")) {
        const auto mode = lowercase_ascii(json_string_value(predicate, "match_mode"));
        const auto pattern = normalize_manifest_token_text(fuzzy_pattern_from_predicate(predicate));
        key += std::to_string(mode.size()) + ":" + mode + std::to_string(pattern.size()) + ":" + pattern;
    }
    return key;
}

template <typename NgramIndexT>
QueryManifestRowResult expand_manifest_fuzzy_labels_impl(const json& record,
                                                         const NgramIndexT& ngram_index,
                                                         const GraphLoadConfig& config,
                                                         uint64_t line_number)
{
    const auto predicates_it = record.find("predicates");
    if (predicates_it == record.end() || !predicates_it->is_array()) {
        fail_graph("query manifest line " + std::to_string(line_number) +
                     " fuzzy query has no predicates array");
    }
    struct FuzzyOccurrence {
        uint32_t label = 0;
        uint8_t source_tier = static_cast<uint8_t>(QueryLabelSourceTier::Fallback);
        uint32_t relevance_rank = 0;
        uint32_t predicate_position = 0;
    };
    std::vector<FuzzyOccurrence> occurrences;
    uint32_t predicate_position = 0;
    for (const json& predicate : *predicates_it) {
        if (active_query_manifest_profile != nullptr) {
            active_query_manifest_profile->fuzzy_predicates += 1;
        }
        std::string mode = json_string_value(predicate, "match_mode");
        if (mode.empty()) {
            mode = json_string_value(predicate, "fuzzy_match_mode");
        }
        const std::string pattern = fuzzy_pattern_from_predicate(predicate);
        if (pattern.empty()) {
            fail_graph("query manifest line " + std::to_string(line_number) +
                         " fuzzy predicate is missing pattern text");
        }
        std::vector<uint8_t> expanded_tiers;
        std::vector<uint32_t> expanded_ranks;
        std::vector<uint32_t> expanded =
            ngram_index.expand(pattern,
                               mode,
                               config.query_manifest_min_ngram_score,
                               std::numeric_limits<uint32_t>::max(),
                               &expanded_tiers,
                               &expanded_ranks);
        if (expanded_tiers.size() != expanded.size() ||
            expanded_ranks.size() != expanded.size()) {
            fail_graph("query manifest fuzzy occurrence metadata is not label-aligned");
        }
        for (size_t idx = 0; idx < expanded.size(); ++idx) {
            if (expanded_tiers[idx] > static_cast<uint8_t>(QueryLabelSourceTier::Fallback)) {
                fail_graph("query manifest fuzzy occurrence has an invalid source tier");
            }
            if (expanded_tiers[idx] ==
                static_cast<uint8_t>(QueryLabelSourceTier::Fallback)) {
                fail_graph(
                    "manifest fuzzy expansion produced a non-literal fallback label");
            }
        }
        // Literal operator output is already sorted and unique. A single
        // predicate needs no occurrence union, sorting or payload copies.
        const auto finalize_start = LoadClock::now();
        const bool ordered_unique = std::adjacent_find(expanded.begin(), expanded.end(),
            std::greater_equal<uint32_t>()) == expanded.end();
        if (predicates_it->size() == 1 && !expanded.empty() && ordered_unique &&
            snapshot_query_preparation_config().optimized) {
            QueryManifestRowResult result;
            result.labels = std::move(expanded);
            result.source_tiers = std::move(expanded_tiers);
            result.relevance_ranks = std::move(expanded_ranks);
            if (active_query_manifest_profile != nullptr)
                active_query_manifest_profile->fuzzy_row_finalize_ns += elapsed_load_ns(finalize_start);
            return result;
        }
        for (size_t idx = 0; idx < expanded.size(); ++idx) {
            occurrences.push_back({expanded[idx], expanded_tiers[idx], expanded_ranks[idx],
                                   predicate_position});
        }
        ++predicate_position;
    }
    const auto finalize_start = LoadClock::now();
    std::sort(occurrences.begin(), occurrences.end(), [](const auto& left, const auto& right) {
        if (left.label != right.label) {
            return left.label < right.label;
        }
        if (left.source_tier != right.source_tier) {
            return left.source_tier < right.source_tier;
        }
        if (left.relevance_rank != right.relevance_rank) {
            return left.relevance_rank < right.relevance_rank;
        }
        return left.predicate_position < right.predicate_position;
    });
    occurrences.erase(
        std::unique(occurrences.begin(), occurrences.end(), [](const auto& left, const auto& right) {
            return left.label == right.label;
        }),
        occurrences.end());
    if (occurrences.empty()) {
        fail_graph("query manifest line " + std::to_string(line_number) +
                     " fuzzy query expanded to zero token labels");
    }
    QueryManifestRowResult result;
    result.labels.reserve(occurrences.size());
    result.source_tiers.reserve(occurrences.size());
    result.relevance_ranks.reserve(occurrences.size());
    for (const FuzzyOccurrence& occurrence : occurrences) {
        result.labels.push_back(occurrence.label);
        result.source_tiers.push_back(occurrence.source_tier);
        result.relevance_ranks.push_back(occurrence.relevance_rank);
    }
    if (active_query_manifest_profile != nullptr) {
        active_query_manifest_profile->fuzzy_row_finalize_ns += elapsed_load_ns(finalize_start);
    }
    return result;
}

QueryManifestRowResult expand_manifest_fuzzy_labels(const json& record,
                                                    const ManifestNgramIndex& ngram_index,
                                                    const GraphLoadConfig& config,
                                                    uint64_t line_number)
{
    return expand_manifest_fuzzy_labels_impl(record, ngram_index, config, line_number);
}

QueryManifestRowResult expand_manifest_fuzzy_labels(const json& record,
                                                    const GlobalTokenNgramIndex& ngram_index,
                                                    const GraphLoadConfig& config,
                                                    uint64_t line_number)
{
    return expand_manifest_fuzzy_labels_impl(record, ngram_index, config, line_number);
}

std::string format_label_vector(const std::vector<uint32_t>& labels)
{
    std::ostringstream output;
    output << '[';
    for (size_t idx = 0; idx < labels.size(); ++idx) {
        if (idx != 0) {
            output << ',';
        }
        output << labels[idx];
    }
    output << ']';
    return output.str();
}

QueryManifestRowResult expand_manifest_fuzzy_labels(const json& record,
                                                    const GlobalTokenOperatorIndex& operator_index,
                                                    const GraphLoadConfig& config,
                                                    uint64_t line_number)
{
    QueryManifestRowResult result =
        expand_manifest_fuzzy_labels_impl(record, operator_index, config, line_number);
    if (!operator_index.parity_check_enabled()) {
        return result;
    }

    const auto parity_start = LoadClock::now();
    QueryManifestRowResult baseline;
    {
        ScopedQueryManifestProfile disable_profile(nullptr);
        baseline = expand_manifest_fuzzy_labels_impl(
            record, operator_index.baseline_index(), config, line_number);
    }
    const uint64_t parity_ns = elapsed_load_ns(parity_start);
    if (active_query_manifest_profile != nullptr) {
        active_query_manifest_profile->operator_parity_check_ns += parity_ns;
        active_query_manifest_profile->operator_parity_rows += 1;
    }
    if (result.labels != baseline.labels) {
        fail_graph("query manifest operator parity mismatch on line " +
                     std::to_string(line_number) + ": operator=" +
                     format_label_vector(result.labels) + " baseline=" +
                     format_label_vector(baseline.labels));
    }
    return result;
}

FuzzyManifestRewriteTestResult evaluate_fuzzy_operator_manifest_row_for_test(
    const std::string& token_ngram_index_path,
    const std::string& operator_index_path,
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& match_modes,
    float min_score,
    uint32_t max_terms,
    const std::string& fill_mode)
{
    if (patterns.empty() || patterns.size() != match_modes.size()) {
        fail_graph("fuzzy manifest rewrite test requires aligned nonempty predicates");
    }
    validate_query_manifest_operator_rewrite_options(fill_mode, false);
    GlobalTokenNgramIndex ngram_index(token_ngram_index_path, true);
    GlobalTokenOperatorIndex operator_index(
        operator_index_path,
        ngram_index,
        false,
        *parse_query_manifest_operator_fill_mode(fill_mode));

    json record;
    record["predicates"] = json::array();
    for (size_t idx = 0; idx < patterns.size(); ++idx) {
        record["predicates"].push_back(
            {{"pattern", patterns[idx]}, {"match_mode", match_modes[idx]}});
    }
    GraphLoadConfig config;
    config.query_manifest_min_ngram_score = min_score;
    config.query_manifest_max_fuzzy_terms = max_terms;
    const QueryManifestRowResult row =
        expand_manifest_fuzzy_labels(record, operator_index, config, 1);

    FuzzyManifestRewriteTestResult result;
    result.labels = row.labels;
    result.source_tiers = row.source_tiers;
    result.relevance_ranks = row.relevance_ranks;
    return result;
}

template <typename NgramIndexT>
void expand_query_manifest_fuzzy_rows_parallel(const std::vector<QueryManifestRowWork>& rows,
                                               const NgramIndexT& ngram_index,
                                               const GraphLoadConfig& config,
                                               std::vector<QueryManifestRowResult>& results,
                                               QueryManifestLoadProfile& profile)
{
    const size_t worker_count = query_manifest_fuzzy_thread_count(rows.size());
    profile.fuzzy_expand_threads = worker_count;
    if (rows.empty()) {
        return;
    }
    const auto wall_start = LoadClock::now();
    if (worker_count <= 1) {
        ScopedQueryManifestProfile scoped_profile(&profile);
        for (size_t row_idx = 0; row_idx < rows.size(); ++row_idx) {
            const auto row_start = LoadClock::now();
            results[row_idx] = expand_manifest_fuzzy_labels(rows[row_idx].record,
                                                            ngram_index,
                                                            config,
                                                            rows[row_idx].line_number);
            profile.fuzzy_expand_ns += elapsed_load_ns(row_start);
            profile.max_final_row_labels =
                std::max<uint64_t>(profile.max_final_row_labels, results[row_idx].labels.size());
            profile.final_row_labels += results[row_idx].labels.size();
        }
        profile.fuzzy_expand_wall_ns += elapsed_load_ns(wall_start);
        return;
    }

    std::vector<QueryManifestLoadProfile> worker_profiles(worker_count);
    std::vector<std::exception_ptr> worker_errors(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const size_t rows_per_worker = rows.size() / worker_count;
    const size_t remainder = rows.size() % worker_count;
    for (size_t worker_idx = 0; worker_idx < worker_count; ++worker_idx) {
        const size_t begin = rows_per_worker * worker_idx + std::min(worker_idx, remainder);
        const size_t end = begin + rows_per_worker + (worker_idx < remainder ? 1 : 0);
        workers.emplace_back([&, worker_idx, begin, end] {
            try {
                ScopedQueryManifestProfile scoped_profile(&worker_profiles[worker_idx]);
                for (size_t row_idx = begin; row_idx < end; ++row_idx) {
                    const auto row_start = LoadClock::now();
                    results[row_idx] = expand_manifest_fuzzy_labels(
                        rows[row_idx].record,
                        ngram_index,
                        config,
                        rows[row_idx].line_number);
                    worker_profiles[worker_idx].fuzzy_expand_ns += elapsed_load_ns(row_start);
                    worker_profiles[worker_idx].max_final_row_labels =
                        std::max<uint64_t>(worker_profiles[worker_idx].max_final_row_labels,
                                           results[row_idx].labels.size());
                    worker_profiles[worker_idx].final_row_labels += results[row_idx].labels.size();
                }
            } catch (...) {
                worker_errors[worker_idx] = std::current_exception();
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    for (const std::exception_ptr& error : worker_errors) {
        if (error) {
            std::rethrow_exception(error);
        }
    }
    for (QueryManifestLoadProfile& worker_profile : worker_profiles) {
        merge_query_manifest_profile(profile, worker_profile);
    }
    profile.fuzzy_expand_wall_ns += elapsed_load_ns(wall_start);
    profile.fuzzy_expand_threads = worker_count;
}

void append_manifest_label_row(QueryManifestLabels& output,
                               QueryManifestRowResult row,
                               QueryLabelOp op,
                               uint32_t label_count)
{
    if (row.labels.empty()) {
        fail_graph("query manifest produced an empty label row");
    }
    for (uint32_t label : row.labels) {
        if (label >= label_count) {
            fail_graph("query manifest produced label out of range");
        }
    }
    if (row.source_tiers.empty() && row.relevance_ranks.empty()) {
        row.source_tiers.assign(
            row.labels.size(), static_cast<uint8_t>(QueryLabelSourceTier::Exact));
        row.relevance_ranks.assign(row.labels.size(), 0);
    }
    if (row.source_tiers.size() != row.labels.size() ||
        row.relevance_ranks.size() != row.labels.size()) {
        fail_graph("query manifest occurrence metadata is not aligned with its label row");
    }
    output.labels.indices.insert(
        output.labels.indices.end(), row.labels.begin(), row.labels.end());
    output.labels.source_tiers.insert(output.labels.source_tiers.end(),
                                      row.source_tiers.begin(),
                                      row.source_tiers.end());
    output.labels.relevance_ranks.insert(output.labels.relevance_ranks.end(),
                                         row.relevance_ranks.begin(),
                                         row.relevance_ranks.end());
    output.labels.indptr.push_back(output.labels.indices.size());
    output.labels.row_labels.push_back(std::move(row.labels));
    output.labels.rows += 1;
    output.ops.push_back(op);
}

std::string hybrid_manifest_context(const json& record, uint64_t line_number)
{
    const std::string query_id = json_string_value(record, "query_id");
    return "query manifest line " + std::to_string(line_number) +
           (query_id.empty() ? std::string() : " query_id=" + query_id);
}

bool json_has_number(const json& value, const char* key)
{
    const auto it = value.find(key);
    return it != value.end() && it->is_number();
}

QueryLabelOp parse_hybrid_top_op(const json& record, uint64_t line_number)
{
    const std::string context = hybrid_manifest_context(record, line_number);
    const auto predicate_it = record.find("predicate");
    if (predicate_it == record.end() || !predicate_it->is_object()) {
        fail_graph(context + " has no predicate object");
    }
    const auto wrapper_it = record.find("predicates");
    if (wrapper_it != record.end()) {
        if (!wrapper_it->is_array() || wrapper_it->size() != 1 ||
            (*wrapper_it)[0] != *predicate_it) {
            fail_graph(context + " has a mismatched predicates wrapper");
        }
    }
    const std::string predicate_op = lowercase_ascii(json_string_value(*predicate_it, "op"));
    const std::string hybrid_logic = lowercase_ascii(json_string_value(record, "hybrid_top_logic"));
    const std::string predicate_logic = lowercase_ascii(json_string_value(record, "predicate_logic"));
    if ((predicate_op != "and" && predicate_op != "or") ||
        predicate_op != hybrid_logic || predicate_op != predicate_logic) {
        fail_graph(context + " has inconsistent hybrid top-level boolean fields");
    }
    return predicate_op == "and" ? QueryLabelOp::And : QueryLabelOp::Or;
}

void validate_hybrid_range_atom(const json& atom,
                                const std::string& subtype,
                                const std::string& context)
{
    if (atom.contains("similarity")) {
        fail_graph(context + " range atom contains unsupported similarity metadata");
    }
    const auto predicates_it = atom.find("predicates");
    if (predicates_it == atom.end() || !predicates_it->is_array()) {
        fail_graph(context + " range atom has no predicates array");
    }
    const size_t expected_terms = subtype == "width_height" ? 2 : 1;
    if ((subtype != "width" && subtype != "height" && subtype != "width_height") ||
        predicates_it->size() != expected_terms) {
        fail_graph(context + " has unsupported or malformed range subtype " + subtype);
    }
    std::array<uint8_t, 2> seen_columns{0, 0};
    for (const json& predicate : *predicates_it) {
        const auto column_it = predicate.find("column");
        if (column_it == predicate.end() || !column_it->is_number_integer() ||
            column_it->get<int64_t>() < 0) {
            fail_graph(context + " range predicate has no nonnegative integer column");
        }
        const uint64_t column = column_it->get<uint64_t>();
        if (column > 1 || seen_columns[column] != 0) {
            fail_graph(context + " range predicate has an invalid or duplicate column");
        }
        seen_columns[column] = 1;
        const std::string op = lowercase_ascii(json_string_value(predicate, "op"));
        if (op == "gt") {
            if (!json_has_number(predicate, "lo") || json_has_number(predicate, "hi")) {
                fail_graph(context + " gt range predicate has malformed bounds");
            }
        } else if (op == "lt") {
            if (!json_has_number(predicate, "hi") || json_has_number(predicate, "lo")) {
                fail_graph(context + " lt range predicate has malformed bounds");
            }
        } else if (op == "between") {
            if (!json_has_number(predicate, "lo") || !json_has_number(predicate, "hi") ||
                !predicate.value("inclusive_min", false) ||
                !predicate.value("inclusive_max", false)) {
                fail_graph(context + " between range predicate has malformed inclusive bounds");
            }
        } else {
            fail_graph(context + " range predicate has unsupported op " + op);
        }
    }
    if ((subtype == "width" && seen_columns[0] == 0) ||
        (subtype == "height" && seen_columns[1] == 0) ||
        (subtype == "width_height" && (seen_columns[0] == 0 || seen_columns[1] == 0))) {
        fail_graph(context + " range subtype does not match its predicate columns");
    }
}

QueryManifestRowWork parse_hybrid_manifest_row(const json& record,
                                               uint32_t label_count,
                                               uint64_t line_number)
{
    QueryManifestRowWork row;
    row.line_number = line_number;
    row.op = parse_hybrid_top_op(record, line_number);
    row.hybrid = true;
    const std::string context = hybrid_manifest_context(record, line_number);
    const json& predicate = record.at("predicate");
    const auto children_it = predicate.find("children");
    if (children_it == predicate.end() || !children_it->is_array() ||
        children_it->size() != 2) {
        fail_graph(context + " must contain exactly two atomic children");
    }

    std::vector<std::string> atom_types;
    atom_types.reserve(2);
    row.hybrid_atoms.reserve(2);
    for (size_t child_idx = 0; child_idx < children_it->size(); ++child_idx) {
        const json& child = (*children_it)[child_idx];
        if (!child.is_object()) {
            fail_graph(context + " contains a non-object child");
        }
        const std::string family = lowercase_ascii(json_string_value(child, "family"));
        const std::string subtype = lowercase_ascii(json_string_value(child, "subtype"));
        HybridManifestAtomWork atom;
        atom.branch_id = row.op == QueryLabelOp::Or
            ? static_cast<uint32_t>(child_idx)
            : 0U;
        if (family == "categorical" && subtype == "single") {
            atom.kind = HybridManifestAtomKind::Label;
            atom.result.labels = extract_manifest_token_ids(child, label_count, line_number);
            if (atom.result.labels.size() != 1) {
                fail_graph(context + " categorical.single child must contain one token label");
            }
            atom_types.push_back("categorical.single");
        } else if (family == "string" && subtype == "exact") {
            atom.kind = HybridManifestAtomKind::Label;
            atom.result.labels = extract_manifest_token_ids(child, label_count, line_number);
            if (atom.result.labels.size() != 1) {
                fail_graph(context + " string.exact child must contain one token label");
            }
            atom_types.push_back("string.exact");
        } else if (family == "string" && subtype == "fuzzy") {
            const auto predicates_it = child.find("predicates");
            if (predicates_it == child.end() || !predicates_it->is_array() ||
                predicates_it->size() != 1) {
                fail_graph(context + " string.fuzzy child must contain one predicate");
            }
            const json& fuzzy_predicate = (*predicates_it)[0];
            std::string mode = lowercase_ascii(json_string_value(fuzzy_predicate, "match_mode"));
            if (mode.empty()) {
                mode = lowercase_ascii(json_string_value(fuzzy_predicate, "fuzzy_match_mode"));
            }
            if (mode != "prefix" && mode != "suffix" && mode != "contains") {
                fail_graph(context + " string.fuzzy child has unsupported match mode " + mode);
            }
            if (fuzzy_pattern_from_predicate(fuzzy_predicate).empty()) {
                fail_graph(context + " string.fuzzy child has no pattern");
            }
            atom.kind = HybridManifestAtomKind::Fuzzy;
            row.fuzzy = true;
            atom_types.push_back("string.fuzzy");
        } else if (family == "range") {
            validate_hybrid_range_atom(child, subtype, context);
            atom.kind = HybridManifestAtomKind::Range;
            atom_types.push_back("range");
        } else {
            fail_graph(context + " has unsupported atomic child " + family + "." + subtype);
        }
        row.hybrid_atoms.push_back(std::move(atom));
    }

    std::sort(atom_types.begin(), atom_types.end());
    const std::string pair = atom_types[0] + "+" + atom_types[1];
    static const std::array<const char*, 5> supported_pairs{
        "categorical.single+range",
        "categorical.single+string.exact",
        "categorical.single+string.fuzzy",
        "range+string.exact",
        "range+string.fuzzy",
    };
    if (std::find(supported_pairs.begin(), supported_pairs.end(), pair) ==
        supported_pairs.end()) {
        fail_graph(context + " has unsupported hybrid atom pair " + pair);
    }
    return row;
}

void normalize_manifest_row_result(QueryManifestRowResult& row)
{
    if (row.source_tiers.empty() && row.relevance_ranks.empty()) {
        row.source_tiers.assign(
            row.labels.size(), static_cast<uint8_t>(QueryLabelSourceTier::Exact));
        row.relevance_ranks.assign(row.labels.size(), 0);
    }
    if (row.source_tiers.size() != row.labels.size() ||
        row.relevance_ranks.size() != row.labels.size()) {
        fail_graph("hybrid manifest occurrence metadata is not label-aligned");
    }
}

QueryManifestRowResult merge_hybrid_label_results(const QueryManifestRowResult& left,
                                                  const QueryManifestRowResult& right,
                                                  bool ordered_inputs = false)
{
    const auto sorted_unique = [](const QueryManifestRowResult& row) {
        return row.source_tiers.size() == row.labels.size() &&
               row.relevance_ranks.size() == row.labels.size() &&
               std::adjacent_find(row.labels.begin(), row.labels.end(),
                   std::greater_equal<uint32_t>()) == row.labels.end();
    };
    if (ordered_inputs && sorted_unique(left) && sorted_unique(right)) {
        QueryManifestRowResult merged;
        const size_t capacity = left.labels.size() + right.labels.size();
        merged.labels.reserve(capacity);
        merged.source_tiers.reserve(capacity);
        merged.relevance_ranks.reserve(capacity);
        const auto append = [&](const QueryManifestRowResult& row, size_t i) {
            merged.labels.push_back(row.labels[i]);
            merged.source_tiers.push_back(row.source_tiers[i]);
            merged.relevance_ranks.push_back(row.relevance_ranks[i]);
        };
        size_t i = 0, j = 0;
        while (i < left.labels.size() && j < right.labels.size()) {
            if (left.labels[i] < right.labels[j]) append(left, i++);
            else if (right.labels[j] < left.labels[i]) append(right, j++);
            else {
                const bool choose_left = left.source_tiers[i] < right.source_tiers[j] ||
                    (left.source_tiers[i] == right.source_tiers[j] &&
                     left.relevance_ranks[i] <= right.relevance_ranks[j]);
                if (choose_left) append(left, i); else append(right, j);
                ++i; ++j;
            }
        }
        while (i < left.labels.size()) append(left, i++);
        while (j < right.labels.size()) append(right, j++);
        return merged;
    }
    struct Occurrence {
        uint32_t label;
        uint8_t tier;
        uint32_t rank;
    };
    std::vector<Occurrence> occurrences;
    occurrences.reserve(left.labels.size() + right.labels.size());
    const auto append = [&](const QueryManifestRowResult& source) {
        for (size_t idx = 0; idx < source.labels.size(); ++idx) {
            occurrences.push_back(
                {source.labels[idx], source.source_tiers[idx], source.relevance_ranks[idx]});
        }
    };
    append(left);
    append(right);
    std::sort(occurrences.begin(), occurrences.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.label != rhs.label) return lhs.label < rhs.label;
        if (lhs.tier != rhs.tier) return lhs.tier < rhs.tier;
        return lhs.rank < rhs.rank;
    });
    QueryManifestRowResult merged;
    for (const Occurrence& occurrence : occurrences) {
        if (!merged.labels.empty() && merged.labels.back() == occurrence.label) {
            continue;
        }
        merged.labels.push_back(occurrence.label);
        merged.source_tiers.push_back(occurrence.tier);
        merged.relevance_ranks.push_back(occurrence.rank);
    }
    return merged;
}

struct HybridClauseBuild {
    QueryManifestRowResult labels;
    uint32_t top_branch_id = 0;
    bool uses_range = false;
};

std::vector<HybridClauseBuild> hybrid_atom_alternatives(HybridManifestAtomWork atom)
{
    if (atom.kind == HybridManifestAtomKind::Range) {
        return {{QueryManifestRowResult{}, atom.branch_id, true}};
    }
    normalize_manifest_row_result(atom.result);
    if (atom.result.labels.empty()) {
        fail_graph("hybrid manifest atom produced no token labels");
    }
    if (atom.kind == HybridManifestAtomKind::Label) {
        return {{std::move(atom.result), atom.branch_id, false}};
    }
    std::vector<HybridClauseBuild> alternatives;
    alternatives.reserve(atom.result.labels.size());
    for (size_t idx = 0; idx < atom.result.labels.size(); ++idx) {
        QueryManifestRowResult singleton;
        singleton.labels.push_back(atom.result.labels[idx]);
        singleton.source_tiers.push_back(atom.result.source_tiers[idx]);
        singleton.relevance_ranks.push_back(atom.result.relevance_ranks[idx]);
        alternatives.push_back({std::move(singleton), atom.branch_id, false});
    }
    return alternatives;
}

void append_hybrid_clause(HybridQueryClauseTable& table, HybridClauseBuild clause)
{
    normalize_manifest_row_result(clause.labels);
    table.labels.insert(table.labels.end(), clause.labels.labels.begin(), clause.labels.labels.end());
    table.source_tiers.insert(table.source_tiers.end(),
                              clause.labels.source_tiers.begin(),
                              clause.labels.source_tiers.end());
    table.relevance_ranks.insert(table.relevance_ranks.end(),
                                 clause.labels.relevance_ranks.begin(),
                                 clause.labels.relevance_ranks.end());
    table.label_indptr.push_back(table.labels.size());
    table.top_branch_ids.push_back(clause.top_branch_id);
    table.uses_original_range.push_back(clause.uses_range ? 1U : 0U);
}

void append_hybrid_manifest_row(QueryManifestLabels& output,
                                QueryManifestRowWork row,
                                uint32_t label_count)
{
    if (!row.hybrid || row.hybrid_atoms.size() != 2) {
        fail_graph("internal hybrid manifest row is malformed");
    }
    std::vector<HybridClauseBuild> left =
        hybrid_atom_alternatives(std::move(row.hybrid_atoms[0]));
    std::vector<HybridClauseBuild> right =
        hybrid_atom_alternatives(std::move(row.hybrid_atoms[1]));
    std::vector<HybridClauseBuild> clauses;
    if (row.op == QueryLabelOp::And) {
        clauses.reserve(left.size() * right.size());
        for (const HybridClauseBuild& lhs : left) {
            for (const HybridClauseBuild& rhs : right) {
                clauses.push_back({merge_hybrid_label_results(lhs.labels, rhs.labels),
                                   0,
                                   lhs.uses_range || rhs.uses_range});
            }
        }
    } else {
        clauses.reserve(left.size() + right.size());
        clauses.insert(clauses.end(),
                       std::make_move_iterator(left.begin()),
                       std::make_move_iterator(left.end()));
        clauses.insert(clauses.end(),
                       std::make_move_iterator(right.begin()),
                       std::make_move_iterator(right.end()));
    }
    if (clauses.empty()) {
        fail_graph("hybrid manifest row lowered to zero DNF clauses");
    }

    // Build the row-level union in one sort.  Re-merging the growing union for
    // every DNF clause turns fuzzy OR rows into quadratic work before the
    // planner can apply its cap (the 50K hybrid workload can have millions of
    // clauses).
    QueryManifestRowResult union_candidates;
    size_t union_size = 0;
    for (const HybridClauseBuild& clause : clauses) {
        union_size += clause.labels.labels.size();
    }
    union_candidates.labels.reserve(union_size);
    union_candidates.source_tiers.reserve(union_size);
    union_candidates.relevance_ranks.reserve(union_size);
    for (const HybridClauseBuild& clause : clauses) {
        union_candidates.labels.insert(union_candidates.labels.end(),
                                       clause.labels.labels.begin(),
                                       clause.labels.labels.end());
        union_candidates.source_tiers.insert(union_candidates.source_tiers.end(),
                                             clause.labels.source_tiers.begin(),
                                             clause.labels.source_tiers.end());
        union_candidates.relevance_ranks.insert(union_candidates.relevance_ranks.end(),
                                                clause.labels.relevance_ranks.begin(),
                                                clause.labels.relevance_ranks.end());
    }
    const QueryManifestRowResult empty_labels;
    QueryManifestRowResult union_labels =
        merge_hybrid_label_results(union_candidates, empty_labels);
    for (const HybridClauseBuild& clause : clauses) {
        append_hybrid_clause(output.clauses, clause);
    }
    if (union_labels.labels.empty()) {
        fail_graph("hybrid manifest row contains no label-bearing atom");
    }
    append_manifest_label_row(output, std::move(union_labels), row.op, label_count);
    output.clauses.finish_clause_row();
    output.clauses.branch_caps.push_back(row.branch_cap);
    output.clauses.branch_cap_is_fuzzy.push_back(row.branch_cap > 0 ? 1U : 0U);
    output.clauses.rows += 1;
    if (!output.clauses.expressions.empty()) output.clauses.expressions.resize(output.clauses.rows);
}

// Supported hybrid rows contain two atoms and at most one fuzzy atom.
// Write their DNF directly into flat arrays: no heap vector for each singleton.
void append_hybrid_manifest_row_flat(QueryManifestLabels& output,
                                     QueryManifestRowWork row, uint32_t label_count, bool lazy_clauses = false)
{
    if (!row.hybrid || row.hybrid_atoms.size() != 2)
        fail_graph("internal hybrid manifest row is malformed");
    for (auto& atom : row.hybrid_atoms) {
        if (atom.kind == HybridManifestAtomKind::Range) continue;
        normalize_manifest_row_result(atom.result);
        if (atom.result.labels.empty()) fail_graph("hybrid manifest atom produced no token labels");
        if (atom.kind == HybridManifestAtomKind::Label && atom.result.labels.size() != 1) {
            append_hybrid_manifest_row(output, std::move(row), label_count);
            return;
        }
    }
    const auto& left = row.hybrid_atoms[0];
    const auto& right = row.hybrid_atoms[1];
    const auto alternatives = [](const auto& atom) -> size_t {
        return atom.kind == HybridManifestAtomKind::Fuzzy ? atom.result.labels.size() : 1;
    };
    if (lazy_clauses) {
        auto& table = output.clauses;
        HybridClauseExpression expression;
        expression.active = true;
        expression.conjunction = row.op == QueryLabelOp::And;
        for (size_t i = 0; i < row.hybrid_atoms.size(); ++i) {
            const auto& atom = row.hybrid_atoms[i];
            auto& reference = expression.atoms[i];
            reference.begin = table.atom_labels.size();
            if (atom.result.labels.size() > UINT32_MAX)
                fail_graph("hybrid atom occurrence count exceeds uint32 range");
            reference.count = static_cast<uint32_t>(atom.result.labels.size());
            reference.range = atom.kind == HybridManifestAtomKind::Range;
            reference.alternatives = atom.kind == HybridManifestAtomKind::Fuzzy;
            expression.branch_ids[i] = atom.branch_id;
            table.atom_labels.insert(table.atom_labels.end(), atom.result.labels.begin(), atom.result.labels.end());
            table.atom_source_tiers.insert(table.atom_source_tiers.end(), atom.result.source_tiers.begin(), atom.result.source_tiers.end());
            table.atom_relevance_ranks.insert(table.atom_relevance_ranks.end(), atom.result.relevance_ranks.begin(), atom.result.relevance_ranks.end());
        }
        table.expressions.resize(table.rows);
        table.expressions.push_back(expression);
    }
    struct Occurrence { uint32_t label; uint8_t tier; uint32_t rank; };
    const auto emit = [&](const HybridManifestAtomWork* lhs, size_t li,
                          const HybridManifestAtomWork* rhs, size_t ri, uint32_t branch) {
        if (lazy_clauses) {
            auto& table = output.clauses;
            table.label_indptr.push_back(table.labels.size());
            table.top_branch_ids.push_back(branch);
            table.uses_original_range.push_back(
                (lhs && lhs->kind == HybridManifestAtomKind::Range) ||
                (rhs && rhs->kind == HybridManifestAtomKind::Range));
            return;
        }
        std::array<Occurrence, 2> values{};
        size_t count = 0;
        bool uses_range = false;
        const auto append = [&](const HybridManifestAtomWork* atom, size_t index) {
            if (!atom) return;
            if (atom->kind == HybridManifestAtomKind::Range) { uses_range = true; return; }
            values[count++] = {atom->result.labels[index], atom->result.source_tiers[index], atom->result.relevance_ranks[index]};
        };
        append(lhs, li); append(rhs, ri);
        const auto less = [](const Occurrence& a, const Occurrence& b) {
            if (a.label != b.label) return a.label < b.label;
            if (a.tier != b.tier) return a.tier < b.tier;
            return a.rank < b.rank;
        };
        if (count == 2 && less(values[1], values[0])) std::swap(values[0], values[1]);
        auto& table = output.clauses;
        for (size_t i = 0; i < count; ++i) {
            if (i && values[i].label == values[i - 1].label) continue;
            table.labels.push_back(values[i].label);
            table.source_tiers.push_back(values[i].tier);
            table.relevance_ranks.push_back(values[i].rank);
        }
        table.label_indptr.push_back(table.labels.size());
        table.top_branch_ids.push_back(branch);
        table.uses_original_range.push_back(uses_range ? 1U : 0U);
    };
    if (lazy_clauses && output.clauses.virtual_descriptors) {
        // The expression completely defines logical IDs, branches and ranges.
        // No per-clause descriptor arrays or expanded loops are required.
    } else if (row.op == QueryLabelOp::And) {
        for (size_t li = 0; li < alternatives(left); ++li)
            for (size_t ri = 0; ri < alternatives(right); ++ri) emit(&left, li, &right, ri, 0);
    } else {
        for (size_t li = 0; li < alternatives(left); ++li) emit(&left, li, nullptr, 0, left.branch_id);
        for (size_t ri = 0; ri < alternatives(right); ++ri) emit(&right, ri, nullptr, 0, right.branch_id);
    }
    auto union_labels = merge_hybrid_label_results(left.result, right.result, true);
    if (union_labels.labels.empty()) fail_graph("hybrid manifest row contains no label-bearing atom");
    append_manifest_label_row(output, std::move(union_labels), row.op, label_count);
    output.clauses.finish_clause_row();
    output.clauses.branch_caps.push_back(row.branch_cap);
    output.clauses.branch_cap_is_fuzzy.push_back(row.branch_cap > 0 ? 1U : 0U);
    ++output.clauses.rows;
    if (!output.clauses.expressions.empty()) output.clauses.expressions.resize(output.clauses.rows);
}

void append_non_hybrid_clause_row(HybridQueryClauseTable& table,
                                  const QueryManifestRowResult& row,
                                  QueryLabelOp op,
                                  uint32_t branch_cap,
                                  bool fuzzy_cap)
{
    if (op == QueryLabelOp::Or) {
        if ((!row.source_tiers.empty() && row.source_tiers.size() != row.labels.size()) ||
            (!row.relevance_ranks.empty() && row.relevance_ranks.size() != row.labels.size())) {
            fail_graph("singleton clause occurrence metadata is not label-aligned");
        }
        if (table.virtual_descriptors) {
            HybridClauseExpression expression;
            expression.active = expression.unary = true;
            auto& atom = expression.atoms[0];
            atom.begin = table.atom_labels.size();
            if (row.labels.size() > UINT32_MAX) fail_graph("singleton alternatives exceed uint32 range");
            atom.count = static_cast<uint32_t>(row.labels.size());
            atom.alternatives = true;
            table.atom_labels.insert(table.atom_labels.end(), row.labels.begin(), row.labels.end());
            for (size_t idx = 0; idx < row.labels.size(); ++idx) {
                table.atom_source_tiers.push_back(row.source_tiers.empty() ? 0 : row.source_tiers[idx]);
                table.atom_relevance_ranks.push_back(row.relevance_ranks.empty() ? 0 : row.relevance_ranks[idx]);
            }
            table.expressions.resize(table.rows);
            table.expressions.push_back(expression);
        } else for (size_t idx = 0; idx < row.labels.size(); ++idx) {
            table.labels.push_back(row.labels[idx]);
            table.source_tiers.push_back(
                row.source_tiers.empty()
                ? static_cast<uint8_t>(QueryLabelSourceTier::Exact)
                : row.source_tiers[idx]);
            table.relevance_ranks.push_back(
                row.relevance_ranks.empty() ? 0 : row.relevance_ranks[idx]);
            table.label_indptr.push_back(table.labels.size());
            table.top_branch_ids.push_back(static_cast<uint32_t>(idx));
            table.uses_original_range.push_back(0);
        }
    }
    table.finish_clause_row();
    table.branch_caps.push_back(branch_cap);
    table.branch_cap_is_fuzzy.push_back(fuzzy_cap && branch_cap > 0 ? 1U : 0U);
    table.rows += 1;
    if (!table.expressions.empty()) table.expressions.resize(table.rows);
}

// One immutable pool for complete fuzzy matches, factorized expressions per
// query, and a single compatibility CSR for consumers of the original union.
// Only retained physical clauses are expanded later by the existing planner.
QueryManifestLabels materialize_shared_manifest_batch(
    const std::vector<QueryManifestRowWork>& rows,
    const std::vector<QueryManifestRowResult>& exact,
    const std::vector<QueryManifestRowResult>& fuzzy,
    uint32_t label_count, uint32_t threads)
{
    QueryManifestLabels output;
    auto& table = output.clauses;
    auto& labels = output.labels;
    table.virtual_descriptors = true;
    table.query_indptr = {0};
    table.label_indptr = {0};
    table.descriptor_indptr = {0};
    table.expressions.reserve(rows.size());
    table.query_indptr.reserve(rows.size() + 1);
    table.descriptor_indptr.reserve(rows.size() + 1);
    table.branch_caps.reserve(rows.size());
    table.branch_cap_is_fuzzy.reserve(rows.size());
    output.ops.reserve(rows.size());
    labels.rows = static_cast<uint32_t>(rows.size());
    labels.cols = label_count;
    labels.indptr.resize(rows.size() + 1, 0);
    size_t atom_values = 0;
    for (const auto& value : fuzzy) atom_values += value.labels.size();
    for (size_t q = 0; q < rows.size(); ++q) {
        for (const auto& atom : rows[q].hybrid_atoms) atom_values += atom.result.labels.size();
        if (!rows[q].hybrid && rows[q].op == QueryLabelOp::Or && rows[q].shared_fuzzy_result == SIZE_MAX)
            atom_values += exact[q].labels.size();
    }
    table.atom_labels.reserve(atom_values);
    table.atom_source_tiers.reserve(atom_values);
    table.atom_relevance_ranks.reserve(atom_values);
    const auto tier = [](const QueryManifestRowResult& row, size_t i) { return row.source_tiers.empty() ? uint8_t{0} : row.source_tiers[i]; };
    const auto rank = [](const QueryManifestRowResult& row, size_t i) { return row.relevance_ranks.empty() ? uint32_t{0} : row.relevance_ranks[i]; };
    const auto append_atom = [&](const QueryManifestRowResult& value) {
        if (value.labels.size() > UINT32_MAX ||
            (!value.source_tiers.empty() && value.source_tiers.size() != value.labels.size()) ||
            (!value.relevance_ranks.empty() && value.relevance_ranks.size() != value.labels.size()))
            fail_graph("shared manifest atom has malformed occurrence metadata");
        HybridClauseAtomReference ref;
        ref.begin = table.atom_labels.size();
        ref.count = static_cast<uint32_t>(value.labels.size());
        for (size_t i = 0; i < value.labels.size(); ++i) {
            if (value.labels[i] >= label_count) fail_graph("query manifest produced label out of range");
            table.atom_labels.push_back(value.labels[i]);
            table.atom_source_tiers.push_back(tier(value, i));
            table.atom_relevance_ranks.push_back(rank(value, i));
        }
        return ref;
    };
    std::vector<HybridClauseAtomReference> fuzzy_refs;
    fuzzy_refs.reserve(fuzzy.size());
    for (const auto& value : fuzzy) fuzzy_refs.push_back(append_atom(value));
    const auto atom_value = [&](const HybridManifestAtomWork& atom) -> const QueryManifestRowResult& {
        return atom.shared_fuzzy_result == SIZE_MAX ? atom.result : fuzzy.at(atom.shared_fuzzy_result);
    };
    const auto row_value = [&](size_t q) -> const QueryManifestRowResult& {
        return rows[q].shared_fuzzy_result == SIZE_MAX ? exact[q] : fuzzy.at(rows[q].shared_fuzzy_result);
    };
    for (size_t q = 0; q < rows.size(); ++q) {
        const auto& row = rows[q];
        HybridClauseExpression expression;
        size_t union_count = 0;
        if (row.hybrid) {
            if (row.hybrid_atoms.size() != 2) fail_graph("shared hybrid requires two typed atoms");
            expression.active = true;
            expression.conjunction = row.op == QueryLabelOp::And;
            for (size_t i = 0; i < 2; ++i) {
                const auto& atom = row.hybrid_atoms[i];
                auto& ref = expression.atoms[i];
                ref = atom.shared_fuzzy_result == SIZE_MAX ? append_atom(atom.result) : fuzzy_refs.at(atom.shared_fuzzy_result);
                ref.range = atom.kind == HybridManifestAtomKind::Range;
                ref.alternatives = atom.kind == HybridManifestAtomKind::Fuzzy;
                expression.branch_ids[i] = atom.branch_id;
            }
            const auto& a = atom_value(row.hybrid_atoms[0]).labels;
            const auto& b = atom_value(row.hybrid_atoms[1]).labels;
            union_count = a.size() + b.size();
            // Supported rows have at most one fuzzy atom; the other contributes
            // zero or one label, so size calculation needs no full union scan.
            const auto& small = a.size() <= b.size() ? a : b;
            const auto& large = a.size() <= b.size() ? b : a;
            for (uint32_t label : small)
                if (std::binary_search(large.begin(), large.end(), label)) --union_count;
        } else {
            const auto& value = row_value(q);
            union_count = value.labels.size();
            if (row.op == QueryLabelOp::Or) {
                expression.active = expression.unary = true;
                expression.atoms[0] = row.shared_fuzzy_result == SIZE_MAX
                    ? append_atom(value) : fuzzy_refs.at(row.shared_fuzzy_result);
                expression.atoms[0].alternatives = true;
            }
        }
        if (!union_count) fail_graph("query manifest produced an empty label row");
        labels.indptr[q + 1] = labels.indptr[q] + union_count;
        table.expressions.push_back(expression);
        table.finish_clause_row();
        table.branch_caps.push_back(row.branch_cap);
        table.branch_cap_is_fuzzy.push_back(row.fuzzy && row.branch_cap > 0);
        ++table.rows;
        output.ops.push_back(row.op);
    }
    labels.indices.resize(labels.indptr.back());
    labels.source_tiers.resize(labels.indptr.back());
    labels.relevance_ranks.resize(labels.indptr.back());
    const size_t workers = go_fann::query_prepare::worker_count(threads, rows.size());
    const size_t chunks = std::min(rows.size(), workers * 4);
    go_fann::query_prepare::parallel_tasks(chunks, static_cast<uint32_t>(workers), [&](size_t chunk) {
        for (size_t q = rows.size() * chunk / chunks; q < rows.size() * (chunk + 1) / chunks; ++q) {
            size_t out = labels.indptr[q];
            const auto append = [&](const QueryManifestRowResult& value, size_t i) {
                if (value.labels[i] >= label_count || out >= labels.indptr[q + 1])
                    fail_graph("shared manifest union has invalid label or size");
                labels.indices[out] = value.labels[i];
                labels.source_tiers[out] = tier(value, i);
                labels.relevance_ranks[out++] = rank(value, i);
            };
            if (!rows[q].hybrid) {
                const auto& value = row_value(q);
                for (size_t i = 0; i < value.labels.size(); ++i) append(value, i);
            } else {
                const auto& a = atom_value(rows[q].hybrid_atoms[0]);
                const auto& b = atom_value(rows[q].hybrid_atoms[1]);
                size_t i = 0, j = 0;
                while (i < a.labels.size() && j < b.labels.size()) {
                    if (a.labels[i] < b.labels[j]) append(a, i++);
                    else if (b.labels[j] < a.labels[i]) append(b, j++);
                    else {
                        if (std::make_pair(tier(a, i), rank(a, i)) <= std::make_pair(tier(b, j), rank(b, j))) append(a, i);
                        else append(b, j);
                        ++i; ++j;
                    }
                }
                while (i < a.labels.size()) append(a, i++);
                while (j < b.labels.size()) append(b, j++);
            }
            if (out != labels.indptr[q + 1]) fail_graph("shared manifest union size mismatch");
        }
    });
    return output;
}

QueryManifestLabels load_query_manifest_jsonl_labels(const GraphLoadConfig& config,
                                                     uint32_t expected_rows,
                                                     uint32_t label_count,
                                                     GraphQueryPreparationTiming* timing)
{
    QueryManifestLoadProfile profile;
    const auto profile_total_start = LoadClock::now();
    ScopedQueryManifestProfile scoped_profile(&profile);
    if (config.query_manifest_max_or_terms == 0) {
        fail_graph("query_manifest_max_or_terms must be positive");
    }
    if (config.query_manifest_max_fuzzy_terms == 0) {
        fail_graph("query_manifest_max_fuzzy_terms must be positive");
    }
    if (!std::isfinite(config.query_manifest_min_ngram_score) ||
        config.query_manifest_min_ngram_score < 0.0f ||
        config.query_manifest_min_ngram_score > 1.0f) {
        fail_graph("query_manifest_min_ngram_score must be finite and in [0, 1]");
    }

    const std::vector<std::string> families = split_csv_lower_set(config.query_manifest_families);
    const std::vector<std::string> subtypes = split_csv_lower_set(config.query_manifest_subtypes);
    std::optional<GlobalTokenNgramIndex> global_ngram_index;
    std::optional<GlobalTokenOperatorIndex> global_operator_index;
    ManifestNgramIndex manifest_ngram_index(config.query_manifest_ngram_size);
    const QueryManifestOperatorRuntimeSelection operator_selection =
        snapshot_query_manifest_operator_selection();
    validate_query_manifest_operator_rewrite_options(
        query_manifest_operator_fill_mode_name(operator_selection.fill_mode),
        operator_selection.parity_check);
    if (operator_selection.parity_check && operator_selection.path.empty()) {
        fail_graph("query manifest operator parity check requires an explicitly configured operator index path");
    }
    if (operator_selection.fill_mode != QueryManifestOperatorFillMode::LegacyFullFallback &&
        operator_selection.path.empty()) {
        fail_graph(
            "query manifest " +
            std::string(query_manifest_operator_fill_mode_name(operator_selection.fill_mode)) +
            " fill requires an explicitly configured operator index path");
    }
    auto stage_start = LoadClock::now();
    const std::optional<std::string> token_ngram_index_path =
        discover_token_ngram_index_path(config.query_manifest_jsonl_path);
    profile.discover_ngram_ns += elapsed_load_ns(stage_start);
    const bool use_global_ngram_index = token_ngram_index_path.has_value();

    QueryManifestLabels output;
    if (use_global_ngram_index) {
        stage_start = LoadClock::now();
        global_ngram_index.emplace(*token_ngram_index_path,
                                   !operator_selection.path.empty());
        profile.global_ngram_load_ns += elapsed_load_ns(stage_start);
        if (global_ngram_index->label_count() != label_count) {
            fail_graph("token ngram label_count=" +
                         std::to_string(global_ngram_index->label_count()) +
                         " does not match runtime label_count=" +
                         std::to_string(label_count) + " in " +
                         *token_ngram_index_path);
        }
        if (!operator_selection.path.empty()) {
            stage_start = LoadClock::now();
            global_operator_index.emplace(operator_selection.path,
                                          *global_ngram_index,
                                          operator_selection.parity_check,
                                          operator_selection.fill_mode);
            profile.operator_index_load_ns += elapsed_load_ns(stage_start);
        }
    } else {
        if (!operator_selection.path.empty()) {
            fail_graph("explicit token operator index " + operator_selection.path +
                         " requires a sibling token_ngram_index.bin and the GlobalTokenNgramIndex path");
        }
        std::ifstream input(config.query_manifest_jsonl_path);
        if (!input) {
            fail_graph("unable to open query manifest JSONL " + config.query_manifest_jsonl_path);
        }
        std::string line;
        uint64_t line_number = 0;
        while (true) {
            stage_start = LoadClock::now();
            const bool has_line = static_cast<bool>(std::getline(input, line));
            profile.fallback_file_read_ns += elapsed_load_ns(stage_start);
            if (!has_line) {
                break;
            }
            ++line_number;
            if (line.empty()) {
                continue;
            }
            json record;
            stage_start = LoadClock::now();
            try {
                record = json::parse(line);
            } catch (const json::parse_error& exc) {
                fail_graph("query manifest JSON parse error on line " +
                             std::to_string(line_number) + ": " + exc.what());
            }
            profile.fallback_index_json_parse_ns += elapsed_load_ns(stage_start);
            stage_start = LoadClock::now();
            add_manifest_token_texts(record, manifest_ngram_index, label_count);
            profile.fallback_index_add_token_ns += elapsed_load_ns(stage_start);
        }
        stage_start = LoadClock::now();
        manifest_ngram_index.finalize();
        profile.fallback_index_finalize_ns += elapsed_load_ns(stage_start);
    }

    const auto query_work_start = LoadClock::now();
    output.labels.cols = label_count;
    output.labels.indptr.push_back(0);
    output.clauses.query_indptr.push_back(0);
    output.clauses.label_indptr.push_back(0);

    uint64_t scanned = 0;
    uint64_t selected = 0;
    uint64_t and_rows = 0;
    uint64_t or_rows = 0;
    uint64_t fuzzy_rows = 0;
    std::vector<QueryManifestRowWork> selected_rows;
    selected_rows.reserve(expected_rows);
    std::vector<QueryManifestRowResult> selected_results;
    selected_results.reserve(expected_rows);
    std::vector<HybridFuzzyTarget> fuzzy_work_to_selected_row;
    fuzzy_work_to_selected_row.reserve(expected_rows);
    std::vector<QueryManifestRowWork> fuzzy_work;
    fuzzy_work.reserve(expected_rows);
    std::vector<QueryManifestRowResult> fuzzy_results;
    fuzzy_results.reserve(expected_rows);

    std::ifstream input(config.query_manifest_jsonl_path);
    if (!input) {
        fail_graph("unable to reopen query manifest JSONL " + config.query_manifest_jsonl_path);
    }
    std::string line;
    uint64_t line_number = 0;
    while (true) {
        stage_start = LoadClock::now();
        const bool has_line = static_cast<bool>(std::getline(input, line));
        profile.file_read_ns += elapsed_load_ns(stage_start);
        if (!has_line) {
            break;
        }
        ++line_number;
        if (line.empty()) {
            ++profile.empty_lines;
            continue;
        }
        json record;
        stage_start = LoadClock::now();
        try {
            record = json::parse(line);
        } catch (const json::parse_error& exc) {
            fail_graph("query manifest JSON parse error on line " +
                         std::to_string(line_number) + ": " + exc.what());
        }
        profile.json_parse_ns += elapsed_load_ns(stage_start);
        ++profile.parsed_records;
        ++scanned;
        stage_start = LoadClock::now();
        if (!manifest_record_allowed(record, families, subtypes)) {
            profile.record_filter_ns += elapsed_load_ns(stage_start);
            ++profile.skipped_records;
            continue;
        }
        profile.record_filter_ns += elapsed_load_ns(stage_start);
        stage_start = LoadClock::now();
        const std::string family = manifest_family(record);
        const std::string subtype = manifest_subtype(record);
        const std::string subtype_key = family + "." + subtype;
        profile.family_subtype_ns += elapsed_load_ns(stage_start);
        QueryManifestRowWork row_work;
        row_work.line_number = line_number;
        std::vector<uint32_t> labels;
        QueryLabelOp op = QueryLabelOp::And;
        if (family == "hybrid") {
            row_work = parse_hybrid_manifest_row(record, label_count, line_number);
            if (row_work.fuzzy) {
                row_work.branch_cap = config.query_manifest_max_fuzzy_terms;
            }
            op = row_work.op;
            if (op == QueryLabelOp::And) {
                ++and_rows;
            } else {
                ++or_rows;
            }
            const size_t selected_row = selected_rows.size();
            const json& children = record.at("predicate").at("children");
            for (size_t atom_idx = 0; atom_idx < row_work.hybrid_atoms.size(); ++atom_idx) {
                if (row_work.hybrid_atoms[atom_idx].kind != HybridManifestAtomKind::Fuzzy) {
                    continue;
                }
                QueryManifestRowWork fuzzy_row;
                fuzzy_row.line_number = line_number;
                fuzzy_row.op = QueryLabelOp::Or;
                fuzzy_row.fuzzy = true;
                fuzzy_row.record = lower_fuzzy_work_record(children.at(atom_idx), line_number);
                fuzzy_work_to_selected_row.push_back({selected_row, atom_idx});
                fuzzy_work.push_back(std::move(fuzzy_row));
                ++fuzzy_rows;
            }
            selected_rows.push_back(std::move(row_work));
            selected_results.push_back(QueryManifestRowResult{});
            ++selected;
            continue;
        } else if ((family == "categorical" && (subtype == "single" || subtype == "and")) ||
            (family == "string" && subtype == "exact")) {
            stage_start = LoadClock::now();
            labels = extract_manifest_token_ids(record, label_count, line_number);
            profile.exact_extract_ns += elapsed_load_ns(stage_start);
            op = QueryLabelOp::And;
            ++and_rows;
            row_work.op = op;
        } else if (family == "categorical" && subtype == "or") {
            stage_start = LoadClock::now();
            labels = extract_manifest_token_ids(record, label_count, line_number);
            profile.exact_extract_ns += elapsed_load_ns(stage_start);
            op = QueryLabelOp::Or;
            ++or_rows;
            row_work.op = op;
            row_work.branch_cap = config.query_manifest_max_or_terms;
        } else if (family == "string" && subtype == "fuzzy") {
            op = QueryLabelOp::Or;
            ++or_rows;
            ++fuzzy_rows;
            row_work.op = op;
            row_work.fuzzy = true;
            row_work.record = lower_fuzzy_work_record(record, line_number);
            fuzzy_work_to_selected_row.push_back({selected_rows.size(), SIZE_MAX});
            fuzzy_work.push_back(std::move(row_work));
            selected_rows.push_back(QueryManifestRowWork{});
            selected_rows.back().line_number = line_number;
            selected_rows.back().op = op;
            selected_rows.back().fuzzy = true;
            selected_rows.back().branch_cap = config.query_manifest_max_fuzzy_terms;
            selected_results.push_back(QueryManifestRowResult{});
            ++selected;
            continue;
        } else if (family == "range") {
            fail_graph("query manifest line " + std::to_string(line_number) +
                         " is range subtype " + subtype_key +
                         "; JSONL label loader currently expects range queries through query_range_path");
        } else {
            fail_graph("query manifest line " + std::to_string(line_number) +
                         " has unsupported subtype " + subtype_key);
        }
        row_work.op = op;
        selected_rows.push_back(std::move(row_work));
        QueryManifestRowResult exact_result;
        exact_result.labels = std::move(labels);
        selected_results.push_back(std::move(exact_result));
        ++selected;
    }

    const auto preparation = snapshot_query_preparation_config();
    const bool shared_atoms = preparation.optimized && config.supports_shared_label_routes;
    std::vector<size_t> fuzzy_aliases(fuzzy_work.size());
    std::iota(fuzzy_aliases.begin(), fuzzy_aliases.end(), size_t{0});
    const auto atom_intern_start = LoadClock::now();
    if (shared_atoms) {
        std::unordered_map<std::string, size_t> atoms;
        atoms.reserve(fuzzy_work.size());
        std::vector<QueryManifestRowWork> unique;
        unique.reserve(fuzzy_work.size());
        for (size_t i = 0; i < fuzzy_work.size(); ++i) {
            const auto [it, inserted] = atoms.emplace(fuzzy_batch_atom_key(fuzzy_work[i]), unique.size());
            fuzzy_aliases[i] = it->second;
            if (inserted) unique.push_back(std::move(fuzzy_work[i]));
        }
        fuzzy_work = std::move(unique);
    }
    profile.fuzzy_expand_wall_ns += elapsed_load_ns(atom_intern_start);
    std::printf("[query-prepare] stage=batch_atoms occurrences=%zu unique=%zu shared=%u online=1\n",
                fuzzy_aliases.size(), fuzzy_work.size(), shared_atoms ? 1U : 0U);
    fuzzy_results.resize(fuzzy_work.size());
    if (!fuzzy_work.empty()) {
        validate_query_manifest_fuzzy_operator_fill_mode(
            query_manifest_operator_fill_mode_name(operator_selection.fill_mode));
        if (global_operator_index.has_value()) {
            expand_query_manifest_fuzzy_rows_parallel(fuzzy_work,
                                                      *global_operator_index,
                                                      config,
                                                      fuzzy_results,
                                                      profile);
        } else if (global_ngram_index.has_value()) {
            expand_query_manifest_fuzzy_rows_parallel(fuzzy_work,
                                                      *global_ngram_index,
                                                      config,
                                                      fuzzy_results,
                                                      profile);
        } else {
            expand_query_manifest_fuzzy_rows_parallel(fuzzy_work,
                                                      manifest_ngram_index,
                                                      config,
                                                      fuzzy_results,
                                                      profile);
        }
        for (size_t idx = 0; idx < fuzzy_aliases.size(); ++idx) {
            const size_t atom_id = fuzzy_aliases[idx];
            const HybridFuzzyTarget target = fuzzy_work_to_selected_row[idx];
            if (target.atom == SIZE_MAX) {
                if (shared_atoms) selected_rows[target.selected_row].shared_fuzzy_result = atom_id;
                else selected_results[target.selected_row] = std::move(fuzzy_results[atom_id]);
            } else {
                if (target.selected_row >= selected_rows.size() ||
                    target.atom >= selected_rows[target.selected_row].hybrid_atoms.size()) {
                    fail_graph("internal hybrid fuzzy expansion target is out of range");
                }
                auto& atom = selected_rows[target.selected_row].hybrid_atoms[target.atom];
                if (shared_atoms) atom.shared_fuzzy_result = atom_id;
                else atom.result = std::move(fuzzy_results[atom_id]);
            }
        }
    }

    stage_start = LoadClock::now();
    const auto materialize = [&](QueryManifestLabels& target, size_t begin, size_t end, bool optimized) {
        target.clauses.virtual_descriptors = optimized && config.supports_shared_label_routes;
        if (target.clauses.virtual_descriptors) target.clauses.descriptor_indptr.assign(1, 0);
        size_t flat_labels = 0, singleton_clauses = 0;
        for (size_t i = begin; i < end; ++i) {
            if (selected_rows[i].hybrid) {
                // Exact reserve for the supported two-atom hybrid shapes.
                size_t clauses = selected_rows[i].op == QueryLabelOp::And ? 1 : 0;
                size_t labels = 0;
                for (const auto& atom : selected_rows[i].hybrid_atoms) {
                    const size_t alternatives = atom.kind == HybridManifestAtomKind::Fuzzy ? atom.result.labels.size() : 1;
                    if (selected_rows[i].op == QueryLabelOp::And) clauses *= alternatives;
                    else clauses += alternatives;
                    labels += atom.result.labels.size();
                }
                singleton_clauses += clauses;
                flat_labels += labels;
            } else {
                flat_labels += selected_results[i].labels.size();
                if (selected_rows[i].op == QueryLabelOp::Or) singleton_clauses += selected_results[i].labels.size();
            }
        }
        target.labels.row_labels.reserve(end - begin);
        target.labels.indptr.reserve(end - begin + 1);
        target.labels.indices.reserve(flat_labels);
        target.labels.source_tiers.reserve(flat_labels);
        target.labels.relevance_ranks.reserve(flat_labels);
        auto& clauses = target.clauses;
        clauses.query_indptr.reserve(end - begin + 1);
        clauses.branch_caps.reserve(end - begin);
        clauses.branch_cap_is_fuzzy.reserve(end - begin);
        clauses.label_indptr.reserve(clauses.virtual_descriptors ? 1 : singleton_clauses + 1);
        if (clauses.virtual_descriptors) {
            clauses.descriptor_indptr.reserve(end - begin + 1);
            clauses.expressions.reserve(end - begin);
            clauses.atom_labels.reserve(flat_labels);
            clauses.atom_source_tiers.reserve(flat_labels);
            clauses.atom_relevance_ranks.reserve(flat_labels);
        }
        if (!config.supports_shared_label_routes || !optimized) {
            clauses.labels.reserve(2 * singleton_clauses);
            clauses.source_tiers.reserve(2 * singleton_clauses);
            clauses.relevance_ranks.reserve(2 * singleton_clauses);
        }
        clauses.top_branch_ids.reserve(clauses.virtual_descriptors ? 0 : singleton_clauses);
        clauses.uses_original_range.reserve(clauses.virtual_descriptors ? 0 : singleton_clauses);
        for (size_t i = begin; i < end; ++i) {
            if (selected_rows[i].hybrid) {
                if (optimized) append_hybrid_manifest_row_flat(target, std::move(selected_rows[i]), label_count, config.supports_shared_label_routes);
                else append_hybrid_manifest_row(target, std::move(selected_rows[i]), label_count);
            } else {
                append_non_hybrid_clause_row(target.clauses, selected_results[i], selected_rows[i].op,
                                             selected_rows[i].branch_cap, selected_rows[i].fuzzy);
                append_manifest_label_row(target, std::move(selected_results[i]), selected_rows[i].op, label_count);
            }
        }
    };
    const size_t prepare_workers = preparation.optimized && selected_rows.size() >= 128
        ? go_fann::query_prepare::worker_count(preparation.threads, selected_rows.size()) : 1;
    if (shared_atoms) {
        output = materialize_shared_manifest_batch(selected_rows, selected_results, fuzzy_results,
                                                   label_count, preparation.threads);
    } else if (prepare_workers <= 1) {
        materialize(output, 0, selected_rows.size(), preparation.optimized);
    } else {
        const size_t count = std::min(selected_rows.size(), prepare_workers * 4);
        std::vector<QueryManifestLabels> chunks(count);
        go_fann::query_prepare::parallel_tasks(count, static_cast<uint32_t>(prepare_workers), [&](size_t i) {
            auto& chunk = chunks[i];
            chunk.labels.cols = label_count;
            chunk.labels.indptr.push_back(0);
            chunk.clauses.query_indptr.push_back(0);
            chunk.clauses.label_indptr.push_back(0);
            materialize(chunk, selected_rows.size() * i / count, selected_rows.size() * (i + 1) / count, true);
        });
        const auto concatenate = [&](auto& destination, auto getter) {
            size_t size = 0;
            for (auto& chunk : chunks) size += getter(chunk).size();
            destination.reserve(size);
            for (auto& chunk : chunks) {
                auto& source = getter(chunk);
                destination.insert(destination.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
            }
        };
        concatenate(output.labels.row_labels, [](auto& chunk) -> auto& { return chunk.labels.row_labels; });
        concatenate(output.labels.indices, [](auto& chunk) -> auto& { return chunk.labels.indices; });
        concatenate(output.labels.source_tiers, [](auto& chunk) -> auto& { return chunk.labels.source_tiers; });
        concatenate(output.labels.relevance_ranks, [](auto& chunk) -> auto& { return chunk.labels.relevance_ranks; });
        concatenate(output.ops, [](auto& chunk) -> auto& { return chunk.ops; });
        concatenate(output.clauses.atom_labels, [](auto& chunk) -> auto& { return chunk.clauses.atom_labels; });
        concatenate(output.clauses.atom_source_tiers, [](auto& chunk) -> auto& { return chunk.clauses.atom_source_tiers; });
        concatenate(output.clauses.atom_relevance_ranks, [](auto& chunk) -> auto& { return chunk.clauses.atom_relevance_ranks; });
        concatenate(output.clauses.labels, [](auto& chunk) -> auto& { return chunk.clauses.labels; });
        concatenate(output.clauses.source_tiers, [](auto& chunk) -> auto& { return chunk.clauses.source_tiers; });
        concatenate(output.clauses.relevance_ranks, [](auto& chunk) -> auto& { return chunk.clauses.relevance_ranks; });
        concatenate(output.clauses.top_branch_ids, [](auto& chunk) -> auto& { return chunk.clauses.top_branch_ids; });
        concatenate(output.clauses.uses_original_range, [](auto& chunk) -> auto& { return chunk.clauses.uses_original_range; });
        concatenate(output.clauses.branch_caps, [](auto& chunk) -> auto& { return chunk.clauses.branch_caps; });
        concatenate(output.clauses.branch_cap_is_fuzzy, [](auto& chunk) -> auto& { return chunk.clauses.branch_cap_is_fuzzy; });
        uint64_t labels = 0, clause_labels = 0, clauses = 0, atom_offset = 0;
        output.clauses.virtual_descriptors = preparation.optimized && config.supports_shared_label_routes;
        if (output.clauses.virtual_descriptors) output.clauses.descriptor_indptr.assign(1, 0);
        uint64_t stored_clauses = 0;
        const bool has_expressions = output.clauses.virtual_descriptors || !output.clauses.atom_labels.empty();
        for (const auto& chunk : chunks) {
            if (has_expressions) {
                for (uint32_t row = 0; row < chunk.clauses.rows; ++row) {
                    auto expression = chunk.clauses.expressions.empty() ? HybridClauseExpression{} : chunk.clauses.expressions[row];
                    if (expression.active) for (auto& atom : expression.atoms) atom.begin += atom_offset;
                    output.clauses.expressions.push_back(expression);
                }
            }
            atom_offset += chunk.clauses.atom_labels.size();
            for (size_t i = 1; i < chunk.labels.indptr.size(); ++i) output.labels.indptr.push_back(labels + chunk.labels.indptr[i]);
            for (size_t i = 1; i < chunk.clauses.query_indptr.size(); ++i) output.clauses.query_indptr.push_back(clauses + chunk.clauses.query_indptr[i]);
            for (size_t i = 1; i < chunk.clauses.label_indptr.size(); ++i) output.clauses.label_indptr.push_back(clause_labels + chunk.clauses.label_indptr[i]);
            labels += chunk.labels.indices.size();
            clause_labels += chunk.clauses.labels.size();
            if (output.clauses.virtual_descriptors) {
                for (size_t i = 1; i < chunk.clauses.descriptor_indptr.size(); ++i)
                    output.clauses.descriptor_indptr.push_back(stored_clauses + chunk.clauses.descriptor_indptr[i]);
                stored_clauses += chunk.clauses.top_branch_ids.size();
            }
            clauses += chunk.clauses.clause_count();
            output.labels.rows += chunk.labels.rows;
            output.clauses.rows += chunk.clauses.rows;
        }
    }
    if (output.clauses.virtual_descriptors && prepare_workers > 1)
        output.clauses.logical_clause_count = output.clauses.query_indptr.back();
    profile.append_row_ns += elapsed_load_ns(stage_start);
    std::printf("[query-prepare] stage=manifest_materialize optimized=%u threads=%zu original_queries=%zu online=1\n",
                preparation.optimized ? 1U : 0U, prepare_workers, selected_rows.size());
    profile.output_nnz = output.labels.indices.size();
    profile.query_work_wall_ns = elapsed_load_ns(query_work_start);

    if (output.labels.rows != expected_rows) {
        fail_graph("query manifest selected row count " +
                     std::to_string(output.labels.rows) +
                     " does not match query vector row count " +
                     std::to_string(expected_rows) +
                     "; provide a query vector file materialized in the same order as the selected JSONL rows");
    }
    if (output.clauses.rows != expected_rows ||
        output.clauses.query_indptr.size() != static_cast<size_t>(expected_rows) + 1 ||
        output.clauses.branch_caps.size() != expected_rows ||
        output.clauses.branch_cap_is_fuzzy.size() != expected_rows ||
        !output.clauses.descriptor_shape_valid() ||
        output.clauses.labels.size() != output.clauses.source_tiers.size() ||
        output.clauses.labels.size() != output.clauses.relevance_ranks.size()) {
        fail_graph("query manifest hybrid clause metadata is not size-aligned");
    }
    std::fprintf(stdout,
                 "[runtime] loaded query manifest JSONL path=%s scanned=%llu selected=%llu and=%llu or=%llu fuzzy=%llu ngram_source=%s ngram_tokens=%zu operator_source=%s operator_tokens=%zu operator_fill_mode=%s operator_parity_check=%d\n",
                 config.query_manifest_jsonl_path.c_str(),
                 static_cast<unsigned long long>(scanned),
                 static_cast<unsigned long long>(selected),
                 static_cast<unsigned long long>(and_rows),
                 static_cast<unsigned long long>(or_rows),
                 static_cast<unsigned long long>(fuzzy_rows),
                 use_global_ngram_index ? "sidecar" : "manifest",
                 use_global_ngram_index ? global_ngram_index->token_count()
                                        : manifest_ngram_index.token_count(),
                 global_operator_index.has_value() ? "configured_sidecar" : "disabled",
                 global_operator_index.has_value() ? global_operator_index->token_count() : 0,
                 query_manifest_operator_fill_mode_name(operator_selection.fill_mode),
                 global_operator_index.has_value() && operator_selection.parity_check ? 1 : 0);
    profile.total_ns = elapsed_load_ns(profile_total_start);
    if (timing != nullptr) {
        timing->query_manifest_prepare_ms = ns_to_ms(profile.query_work_wall_ns);
        timing->query_manifest_file_read_detail_ms = ns_to_ms(profile.file_read_ns);
        timing->query_manifest_json_parse_detail_ms = ns_to_ms(profile.json_parse_ns);
        timing->query_manifest_record_select_detail_ms =
            ns_to_ms(profile.record_filter_ns + profile.family_subtype_ns);
        timing->query_manifest_exact_extract_detail_ms = ns_to_ms(profile.exact_extract_ns);
        timing->query_manifest_fuzzy_expand_wall_detail_ms =
            ns_to_ms(profile.fuzzy_expand_wall_ns);
        timing->query_manifest_row_materialize_detail_ms = ns_to_ms(profile.append_row_ns);
        timing->static_manifest_discovery_ms = ns_to_ms(profile.discover_ngram_ns);
        timing->static_manifest_ngram_load_ms = ns_to_ms(profile.global_ngram_load_ns);
        timing->static_manifest_operator_load_ms = ns_to_ms(profile.operator_index_load_ns);
        timing->static_manifest_fallback_index_build_ms =
            ns_to_ms(profile.fallback_file_read_ns +
                     profile.fallback_index_json_parse_ns +
                     profile.fallback_index_add_token_ns +
                     profile.fallback_index_finalize_ns);
    }
    if (graph_load_timing_enabled()) {
        std::fprintf(stdout,
                     "[runtime-query-manifest-profile] total_ms=%.3f discover_ngram_ms=%.3f global_ngram_load_ms=%.3f fallback_parse_ms=%.3f fallback_add_token_ms=%.3f fallback_finalize_ms=%.3f json_parse_ms=%.3f record_filter_ms=%.3f family_subtype_ms=%.3f exact_extract_ms=%.3f fuzzy_expand_ms=%.3f fuzzy_expand_wall_ms=%.3f fuzzy_expand_threads=%llu fuzzy_row_finalize_ms=%.3f append_row_ms=%.3f expand_normalize_ngram_ms=%.3f expand_posting_lookup_ms=%.3f expand_score_filter_ms=%.3f expand_candidate_sort_ms=%.3f expand_label_finalize_ms=%.3f records=%llu parsed=%llu skipped=%llu empty=%llu fuzzy_rows=%llu fuzzy_predicates=%llu fuzzy_patterns=%llu pattern_chars=%llu grams_total=%llu grams_found=%llu grams_missing=%llu postings_scanned=%llu overlap_candidates=%llu scored_candidates=%llu emitted_labels=%llu final_row_labels=%llu output_nnz=%llu max_grams=%llu max_postings=%llu max_overlap_candidates=%llu max_scored_candidates=%llu max_emitted_labels=%llu max_final_row_labels=%llu max_expand_ms=%.3f\n",
                     ns_to_ms(profile.total_ns),
                     ns_to_ms(profile.discover_ngram_ns),
                     ns_to_ms(profile.global_ngram_load_ns),
                     ns_to_ms(profile.fallback_index_json_parse_ns),
                     ns_to_ms(profile.fallback_index_add_token_ns),
                     ns_to_ms(profile.fallback_index_finalize_ns),
                     ns_to_ms(profile.json_parse_ns),
                     ns_to_ms(profile.record_filter_ns),
                     ns_to_ms(profile.family_subtype_ns),
                     ns_to_ms(profile.exact_extract_ns),
                     ns_to_ms(profile.fuzzy_expand_ns),
                     ns_to_ms(profile.fuzzy_expand_wall_ns),
                     static_cast<unsigned long long>(profile.fuzzy_expand_threads),
                     ns_to_ms(profile.fuzzy_row_finalize_ns),
                     ns_to_ms(profile.append_row_ns),
                     ns_to_ms(profile.expand_normalize_ngram_ns),
                     ns_to_ms(profile.expand_posting_lookup_ns),
                     ns_to_ms(profile.expand_score_filter_ns),
                     ns_to_ms(profile.expand_candidate_sort_ns),
                     ns_to_ms(profile.expand_label_finalize_ns),
                     static_cast<unsigned long long>(scanned),
                     static_cast<unsigned long long>(profile.parsed_records),
                     static_cast<unsigned long long>(profile.skipped_records),
                     static_cast<unsigned long long>(profile.empty_lines),
                     static_cast<unsigned long long>(fuzzy_rows),
                     static_cast<unsigned long long>(profile.fuzzy_predicates),
                     static_cast<unsigned long long>(profile.fuzzy_patterns),
                     static_cast<unsigned long long>(profile.pattern_chars),
                     static_cast<unsigned long long>(profile.grams_total),
                     static_cast<unsigned long long>(profile.grams_found),
                     static_cast<unsigned long long>(profile.grams_missing),
                     static_cast<unsigned long long>(profile.postings_scanned),
                     static_cast<unsigned long long>(profile.overlap_candidates),
                     static_cast<unsigned long long>(profile.scored_candidates),
                     static_cast<unsigned long long>(profile.emitted_labels),
                     static_cast<unsigned long long>(profile.final_row_labels),
                     static_cast<unsigned long long>(profile.output_nnz),
                     static_cast<unsigned long long>(profile.max_grams),
                     static_cast<unsigned long long>(profile.max_postings),
                     static_cast<unsigned long long>(profile.max_overlap_candidates),
                     static_cast<unsigned long long>(profile.max_scored_candidates),
                     static_cast<unsigned long long>(profile.max_emitted_labels),
                     static_cast<unsigned long long>(profile.max_final_row_labels),
                     ns_to_ms(profile.max_expand_ns));
        if (global_operator_index.has_value()) {
            std::fprintf(stdout,
                         "[runtime-fuzzy-operator-profile] fill_mode=%s load_ms=%.3f lookup_ms=%.3f parity_check_ms=%.3f lookups=%llu direct_only_hits=%llu fallbacks=%llu direct_candidates=%llu direct_matches=%llu baseline_grams=%llu baseline_postings=%llu baseline_candidates=%llu parity_rows=%llu\n",
                         query_manifest_operator_fill_mode_name(operator_selection.fill_mode),
                         ns_to_ms(profile.operator_index_load_ns),
                         ns_to_ms(profile.operator_lookup_ns),
                         ns_to_ms(profile.operator_parity_check_ns),
                         static_cast<unsigned long long>(profile.operator_lookups),
                         static_cast<unsigned long long>(profile.operator_direct_only_hits),
                         static_cast<unsigned long long>(profile.operator_fallbacks),
                         static_cast<unsigned long long>(profile.operator_direct_candidates),
                         static_cast<unsigned long long>(profile.operator_direct_matches),
                         static_cast<unsigned long long>(profile.operator_baseline_grams),
                         static_cast<unsigned long long>(profile.operator_baseline_postings),
                         static_cast<unsigned long long>(profile.operator_baseline_candidates),
                         static_cast<unsigned long long>(profile.operator_parity_rows));
        }
        std::fflush(stdout);
    }
    return output;
}

HybridManifestParseTestResult evaluate_hybrid_query_manifest_row_for_test(
    const std::string& json_line,
    const std::vector<uint32_t>& fuzzy_labels,
    const std::vector<uint8_t>& fuzzy_source_tiers,
    const std::vector<uint32_t>& fuzzy_relevance_ranks,
    uint32_t label_count, bool lazy_clauses)
{
    json record;
    try {
        record = json::parse(json_line);
    } catch (const json::parse_error& exc) {
        fail_graph(std::string("hybrid manifest test JSON parse error: ") + exc.what());
    }
    QueryManifestRowWork row = parse_hybrid_manifest_row(record, label_count, 1);
    bool assigned_fuzzy = false;
    for (HybridManifestAtomWork& atom : row.hybrid_atoms) {
        if (atom.kind != HybridManifestAtomKind::Fuzzy) {
            continue;
        }
        if (assigned_fuzzy) {
            fail_graph("hybrid manifest test row contains multiple fuzzy atoms");
        }
        atom.result.labels = fuzzy_labels;
        atom.result.source_tiers = fuzzy_source_tiers;
        atom.result.relevance_ranks = fuzzy_relevance_ranks;
        assigned_fuzzy = true;
    }
    if (assigned_fuzzy != !fuzzy_labels.empty()) {
        fail_graph(assigned_fuzzy
                         ? "hybrid manifest test fuzzy expansion is empty"
                         : "hybrid manifest test supplied fuzzy labels to a non-fuzzy row");
    }
    QueryManifestLabels output;
    output.labels.cols = label_count;
    output.labels.indptr.push_back(0);
    output.clauses.query_indptr.push_back(0);
    output.clauses.label_indptr.push_back(0);
    output.clauses.virtual_descriptors = lazy_clauses && snapshot_query_preparation_config().optimized;
    if (output.clauses.virtual_descriptors) output.clauses.descriptor_indptr.push_back(0);
    const QueryLabelOp top_op = row.op;
    if (snapshot_query_preparation_config().optimized)
        append_hybrid_manifest_row_flat(output, std::move(row), label_count, lazy_clauses);
    else
        append_hybrid_manifest_row(output, std::move(row), label_count);
    return {std::move(output.labels), std::move(output.clauses), top_op};
}

SparseLabelMatrix load_spmat_labels(const std::string& path,
                                    uint32_t row_limit,
                                    bool materialize_row_labels)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open spmat label file " + path);
    }

    uint64_t rows = 0;
    uint64_t cols = 0;
    uint64_t nnz = 0;
    input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    input.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    input.read(reinterpret_cast<char*>(&nnz), sizeof(nnz));
    if (!input) {
        fail_graph("invalid spmat header in " + path);
    }
    if (rows > std::numeric_limits<uint32_t>::max()) {
        fail_graph("spmat row count exceeds uint32 range in " + path);
    }
    if (cols > std::numeric_limits<uint32_t>::max()) {
        fail_graph("spmat label count exceeds uint32 range in " + path);
    }

    const uint32_t effective_rows = (row_limit > 0 && row_limit < rows) ? row_limit : static_cast<uint32_t>(rows);
    SparseLabelMatrix matrix;
    matrix.rows = effective_rows;
    matrix.cols = static_cast<uint32_t>(cols);
    matrix.indptr.resize(static_cast<size_t>(effective_rows) + 1);

    input.read(reinterpret_cast<char*>(matrix.indptr.data()), matrix.indptr.size() * sizeof(uint64_t));
    if (!input) {
        fail_graph("failed to read spmat indptr from " + path);
    }
    if (matrix.indptr.front() != 0) {
        fail_graph("spmat indptr must start at 0 in " + path);
    }
    for (uint32_t row = 0; row < effective_rows; ++row) {
        if (matrix.indptr[row] > matrix.indptr[row + 1]) {
            fail_graph("spmat indptr is not monotonic in " + path);
        }
        if (matrix.indptr[row + 1] > nnz) {
            fail_graph("spmat indptr exceeds nnz in " + path);
        }
    }
    const uint64_t effective_nnz = matrix.indptr[effective_rows];
    if (effective_nnz > std::numeric_limits<size_t>::max()) {
        fail_graph("spmat nnz exceeds host allocation range in " + path);
    }

    const std::streamoff indptr_bytes = static_cast<std::streamoff>((rows + 1) * sizeof(uint64_t));
    const std::streamoff index_offset = static_cast<std::streamoff>(sizeof(uint64_t) * 3) + indptr_bytes;
    input.clear();
    input.seekg(index_offset, std::ios::beg);
    if (!input) {
        fail_graph("failed to seek to spmat indices in " + path);
    }
    matrix.indices.resize(static_cast<size_t>(effective_nnz));
    if (!matrix.indices.empty()) {
        input.read(reinterpret_cast<char*>(matrix.indices.data()),
                   matrix.indices.size() * sizeof(uint32_t));
    }
    if (!input) {
        fail_graph("failed to read spmat indices from " + path);
    }

    if (materialize_row_labels) {
        matrix.row_labels.reserve(effective_rows);
    }

    uint64_t write_cursor = 0;
    for (uint32_t row = 0; row < effective_rows; ++row) {
        const uint64_t begin = matrix.indptr[row];
        const uint64_t end = matrix.indptr[row + 1];
        if (begin > end || end > matrix.indices.size()) {
            fail_graph("spmat row span is invalid in " + path);
        }
        matrix.indptr[row] = write_cursor;
        if (begin == end) {
            if (materialize_row_labels) {
                matrix.row_labels.emplace_back();
            }
            continue;
        }

        uint32_t* row_begin = matrix.indices.data() + static_cast<size_t>(begin);
        uint32_t* row_end = matrix.indices.data() + static_cast<size_t>(end);
        for (uint32_t* it = row_begin; it != row_end; ++it) {
            if (*it >= matrix.cols) {
                fail_graph("spmat label index out of range in " + path);
            }
        }
        std::sort(row_begin, row_end);
        uint32_t* unique_end = std::unique(row_begin, row_end);
        const uint64_t row_nnz = static_cast<uint64_t>(unique_end - row_begin);
        if (write_cursor != begin) {
            std::memmove(matrix.indices.data() + static_cast<size_t>(write_cursor),
                         row_begin,
                         static_cast<size_t>(row_nnz) * sizeof(uint32_t));
        }
        if (materialize_row_labels) {
            matrix.row_labels.emplace_back(matrix.indices.begin() + static_cast<size_t>(write_cursor),
                                          matrix.indices.begin() + static_cast<size_t>(write_cursor + row_nnz));
        }
        write_cursor += row_nnz;
    }
    matrix.indptr[effective_rows] = write_cursor;
    matrix.indices.resize(static_cast<size_t>(write_cursor));
    if (materialize_row_labels) {
        if (matrix.row_labels.size() != effective_rows) {
            fail_graph("failed to materialize sparse label row cache");
        }
    }
    return matrix;
}

QueryRangeTable load_query_range_binary(const std::string& path, uint32_t expected_query_count)
{
    QueryRangeTable ranges;
    ranges.rows = expected_query_count;
    ranges.range_count = 1;
    if (path.empty()) {
        ranges.modes.assign(expected_query_count, 0);
        ranges.lower_bounds.assign(expected_query_count, 0.0f);
        ranges.upper_bounds.assign(expected_query_count, 0.0f);
        return ranges;
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open query range file " + path);
    }

    uint32_t query_count = 0;
    input.read(reinterpret_cast<char*>(&query_count), sizeof(query_count));
    if (!input) {
        fail_graph("invalid query range header in " + path);
    }
    if (query_count != expected_query_count) {
        fail_graph("query range count does not match query vector rows");
    }

    const std::streampos payload_begin = input.tellg();
    input.seekg(0, std::ios::end);
    const std::streamoff payload_bytes = input.tellg() - payload_begin;
    input.seekg(payload_begin);

    constexpr std::streamoff kPackedRecordBytes = sizeof(uint8_t) + sizeof(float) + sizeof(float);
    struct QueryRangeRecordAligned {
        uint8_t mode;
        float upper;
        float lower;
    };
    static_assert(sizeof(QueryRangeRecordAligned) == 12, "aligned query range record layout must be 12 bytes");
    constexpr std::streamoff kAlignedRecordBytes = sizeof(QueryRangeRecordAligned);
    constexpr std::streamoff kHeaderBytesLegacy = sizeof(uint32_t);
    auto load_records = [&](uint32_t range_count, bool aligned) {
        ranges.range_count = range_count;
        ranges.modes.resize(static_cast<size_t>(query_count) * range_count, 0);
        ranges.lower_bounds.resize(static_cast<size_t>(query_count) * range_count, 0.0f);
        ranges.upper_bounds.resize(static_cast<size_t>(query_count) * range_count, 0.0f);
        const size_t total_records = static_cast<size_t>(query_count) * range_count;
        if (aligned) {
            std::vector<QueryRangeRecordAligned> records(total_records);
            input.read(reinterpret_cast<char*>(records.data()), sizeof(QueryRangeRecordAligned) * records.size());
            if (!input) {
                fail_graph("failed to read aligned query range payload from " + path);
            }
            for (size_t i = 0; i < total_records; ++i) {
                ranges.modes[i] = records[i].mode;
                ranges.lower_bounds[i] = records[i].lower;
                ranges.upper_bounds[i] = records[i].upper;
            }
            return;
        }
        for (size_t i = 0; i < total_records; ++i) {
            uint8_t mode = 0;
            float upper = 0.0f;
            float lower = 0.0f;
            input.read(reinterpret_cast<char*>(&mode), sizeof(mode));
            input.read(reinterpret_cast<char*>(&upper), sizeof(upper));
            input.read(reinterpret_cast<char*>(&lower), sizeof(lower));
            if (!input) {
                fail_graph("failed to read packed query range payload from " + path);
            }
            ranges.modes[i] = mode;
            ranges.lower_bounds[i] = lower;
            ranges.upper_bounds[i] = upper;
        }
    };

    if (payload_bytes == static_cast<std::streamoff>(query_count) * kAlignedRecordBytes) {
        load_records(1, true);
    } else if (payload_bytes == static_cast<std::streamoff>(query_count) * kPackedRecordBytes) {
        load_records(1, false);
    } else {
        input.clear();
        input.seekg(kHeaderBytesLegacy, std::ios::beg);
        uint32_t range_count = 0;
        input.read(reinterpret_cast<char*>(&range_count), sizeof(range_count));
        if (!input || range_count == 0) {
            fail_graph("query range payload size does not match packed/aligned layouts for " + path);
        }
        const std::streamoff merged_payload_bytes = payload_bytes - static_cast<std::streamoff>(sizeof(uint32_t));
        const std::streamoff expected_aligned =
            static_cast<std::streamoff>(query_count) * range_count * kAlignedRecordBytes;
        const std::streamoff expected_packed =
            static_cast<std::streamoff>(query_count) * range_count * kPackedRecordBytes;
        if (merged_payload_bytes == expected_aligned) {
            load_records(range_count, true);
        } else if (merged_payload_bytes == expected_packed) {
            load_records(range_count, false);
        } else {
            fail_graph("query range payload size does not match packed/aligned layouts for " + path);
        }
    }
    return ranges;
}

RangeLabelIntervals load_range_label_manifest(const std::string& path, uint32_t expected_label_count)
{
    RangeLabelIntervals intervals;
    intervals.range_count = 1;
    intervals.has_interval.assign(expected_label_count, 0);
    intervals.lower_bounds.assign(expected_label_count, 0.0f);
    intervals.upper_bounds.assign(expected_label_count, 0.0f);
    intervals.level_indices.assign(expected_label_count, UINT32_MAX);
    intervals.bin_indices.assign(expected_label_count, UINT32_MAX);
    intervals.range_indices.assign(expected_label_count, UINT32_MAX);

    if (path.empty()) {
        return intervals;
    }

    std::ifstream input(path);
    if (!input) {
        fail_graph("unable to open range label manifest " + path);
    }

    json root;
    input >> root;
    if (!root.is_object()) {
        fail_graph("range label manifest root must be an object");
    }

    auto note_range_index = [&](uint32_t range_index) {
        if (range_index != UINT32_MAX) {
            intervals.range_count = std::max(intervals.range_count, range_index + 1);
        }
    };

    auto ranges_it = root.find("ranges");
    if (ranges_it != root.end()) {
        if (!ranges_it->is_array()) {
            fail_graph("range label manifest ranges field must be an array");
        }
        for (size_t range_pos = 0; range_pos < ranges_it->size(); ++range_pos) {
            const auto& range_entry = (*ranges_it)[range_pos];
            if (!range_entry.is_object()) {
                continue;
            }
            const uint32_t range_index = range_entry.value("range_index", static_cast<uint32_t>(range_pos));
            note_range_index(range_index);
            auto range_levels_it = range_entry.find("levels");
            if (range_levels_it == range_entry.end() || !range_levels_it->is_array()) {
                continue;
            }
            for (const auto& level : *range_levels_it) {
                if (!level.is_object()) {
                    continue;
                }
                const uint32_t level_index = level.value("level_index", UINT32_MAX);
                auto partitions_it = level.find("partitions");
                if (partitions_it == level.end() || !partitions_it->is_array()) {
                    continue;
                }
                for (const auto& partition : *partitions_it) {
                    if (!partition.is_object()) {
                        continue;
                    }
                    const uint32_t label_id = partition.value("label_id", UINT32_MAX);
                    if (label_id >= expected_label_count) {
                        fail_graph("range label manifest label_id out of range");
                    }
                    auto interval_it = partition.find("interval");
                    if (interval_it == partition.end() || !interval_it->is_array() || interval_it->size() != 2) {
                        fail_graph("range label manifest partition missing interval pair");
                    }
                    intervals.has_interval[label_id] = 1;
                    intervals.lower_bounds[label_id] = (*interval_it)[0].get<float>();
                    intervals.upper_bounds[label_id] = (*interval_it)[1].get<float>();
                    intervals.level_indices[label_id] = level_index;
                    intervals.bin_indices[label_id] = partition.value("bin_index", UINT32_MAX);
                    intervals.range_indices[label_id] = range_index;
                }
            }
        }
        return intervals;
    }

    auto levels_it = root.find("levels");
    if (levels_it == root.end() || !levels_it->is_array()) {
        fail_graph("range label manifest missing levels/ranges array");
    }

    for (const auto& level : *levels_it) {
        if (!level.is_object()) {
            continue;
        }
        const uint32_t level_index = level.value("level_index", UINT32_MAX);
        auto partitions_it = level.find("partitions");
        if (partitions_it == level.end() || !partitions_it->is_array()) {
            continue;
        }
        for (const auto& partition : *partitions_it) {
            if (!partition.is_object()) {
                continue;
            }
            const uint32_t label_id = partition.value("label_id", UINT32_MAX);
            if (label_id >= expected_label_count) {
                fail_graph("range label manifest label_id out of range");
            }
            auto interval_it = partition.find("interval");
            if (interval_it == partition.end() || !interval_it->is_array() || interval_it->size() != 2) {
                fail_graph("range label manifest partition missing interval pair");
            }
            intervals.has_interval[label_id] = 1;
            intervals.lower_bounds[label_id] = (*interval_it)[0].get<float>();
            intervals.upper_bounds[label_id] = (*interval_it)[1].get<float>();
            intervals.level_indices[label_id] = level_index;
            intervals.bin_indices[label_id] = partition.value("bin_index", UINT32_MAX);
            const uint32_t range_index = partition.value("range_index", 0u);
            note_range_index(range_index);
            intervals.range_indices[label_id] = range_index;
        }
    }

    return intervals;
}

Uint32MatrixIbin load_uint32_matrix_ibin(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open graph file " + path);
    }

    Uint32MatrixIbin matrix;
    input.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
    input.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
    if (!input) {
        fail_graph("invalid graph header in " + path);
    }

    matrix.data.resize(static_cast<size_t>(matrix.rows) * matrix.cols);
    input.read(reinterpret_cast<char*>(matrix.data.data()), matrix.data.size() * sizeof(uint32_t));
    if (!input) {
        fail_graph("failed to read graph payload from " + path);
    }
    return matrix;
}

Uint32MatrixIbin load_graph_matrix_auto(const std::string& path)
{
    if (!go_fann::graph_file::has_graph_file_magic(path))
        return load_uint32_matrix_ibin(path);
    auto loaded = go_fann::graph_file::load_graph_file_as_contiguous_matrix(path);
    Uint32MatrixIbin matrix;
    matrix.rows = loaded.rows;
    matrix.cols = loaded.cols;
    matrix.data = std::move(loaded.data);
    return matrix;
}

LowfreqPostingIndex load_lowfreq_posting_index(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        fail_graph("unable to open low-frequency index " + path);
    }

    LowfreqPostingIndex index;
    input.read(reinterpret_cast<char*>(&index.group_count), sizeof(index.group_count));
    input.read(reinterpret_cast<char*>(&index.total_rows), sizeof(index.total_rows));
    if (!input) {
        fail_graph("failed to read low-frequency index header from " + path);
    }

    index.label_size.resize(index.group_count);
    index.label_offset.resize(index.group_count);
    index.sample_ids.resize(index.total_rows);
    index.row_filter_mask.resize(index.total_rows);

    input.read(reinterpret_cast<char*>(index.label_size.data()), index.label_size.size() * sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(index.label_offset.data()), index.label_offset.size() * sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(index.sample_ids.data()), index.sample_ids.size() * sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(index.row_filter_mask.data()), index.row_filter_mask.size() * sizeof(uint64_t));
    if (!input) {
        fail_graph("failed to read low-frequency index payload from " + path);
    }
    return index;
}

void save_lowfreq_posting_index(const std::string& path, const LowfreqPostingIndex& index)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        fail_graph("unable to write low-frequency index " + path);
    }
    output.write(reinterpret_cast<const char*>(&index.group_count), sizeof(index.group_count));
    output.write(reinterpret_cast<const char*>(&index.total_rows), sizeof(index.total_rows));
    output.write(reinterpret_cast<const char*>(index.label_size.data()), index.label_size.size() * sizeof(uint32_t));
    output.write(reinterpret_cast<const char*>(index.label_offset.data()), index.label_offset.size() * sizeof(uint32_t));
    output.write(reinterpret_cast<const char*>(index.sample_ids.data()), index.sample_ids.size() * sizeof(uint32_t));
    output.write(reinterpret_cast<const char*>(index.row_filter_mask.data()), index.row_filter_mask.size() * sizeof(uint64_t));
    if (!output) {
        fail_graph("failed to write low-frequency index " + path);
    }
}

GroupLayout build_identity_group_layout(uint32_t label_count)
{
    GroupLayout layout;
    layout.groups.resize(label_count);
    layout.label_to_group.resize(label_count, UINT32_MAX);
    for (uint32_t label = 0; label < label_count; ++label) {
        layout.groups[label].push_back(label);
        layout.label_to_group[label] = label;
    }
    return layout;
}

GroupLayout load_group_layout(const std::string& path, uint32_t label_count)
{
    if (path.empty()) {
        return build_identity_group_layout(label_count);
    }

    std::ifstream input(path);
    if (!input) {
        fail_graph("unable to open group file " + path);
    }

    GroupLayout layout;
    layout.label_to_group.assign(label_count, UINT32_MAX);

    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream iss(line);
        uint32_t label = 0;
        std::vector<uint32_t> group;
        while (iss >> label) {
            if (label >= label_count) {
                fail_graph("group file label out of range in " + path);
            }
            if (layout.label_to_group[label] != UINT32_MAX) {
                fail_graph("group file repeats label " + std::to_string(label));
            }
            group.push_back(label);
        }
        if (!group.empty()) {
            const uint32_t group_id = static_cast<uint32_t>(layout.groups.size());
            for (uint32_t label_id : group) {
                layout.label_to_group[label_id] = group_id;
            }
            layout.groups.push_back(std::move(group));
        }
    }

    uint32_t autofilled_count = 0;
    for (uint32_t label = 0; label < label_count; ++label) {
        if (layout.label_to_group[label] == UINT32_MAX) {
            const uint32_t group_id = static_cast<uint32_t>(layout.groups.size());
            layout.groups.push_back({label});
            layout.label_to_group[label] = group_id;
            ++autofilled_count;
        }
    }
    if (autofilled_count > 0) {
        std::fprintf(stderr,
                     "[runtime] group file '%s' missing %u labels; defaulted them to singleton groups\n",
                     path.c_str(),
                     autofilled_count);
    }
    return layout;
}

GraphMetadata build_graph_metadata_from_posting_data(const SparseLabelMatrix& data_labels,
                                                        int specificity_threshold,
                                                        const GroupLayout& group_layout,
                                                        const GroupPostingData& posting_data)
{
    GraphMetadata metadata;
    metadata.label_count = static_cast<uint32_t>(group_layout.groups.size());
    metadata.original_label_count = data_labels.cols;
    metadata.specificity_threshold = specificity_threshold;
    metadata.cat_freq.assign(metadata.label_count, 0);
    metadata.label_filter_bit.assign(metadata.original_label_count, 0);
    for (uint32_t group_id = 0; group_id < metadata.label_count; ++group_id) {
        metadata.cat_freq[group_id] = static_cast<uint32_t>(posting_data.sample_ids_by_group[group_id].size());
    }
    metadata.group_posting_size = metadata.cat_freq;

    metadata.label_size.assign(metadata.label_count, 0);
    metadata.label_offset.assign(metadata.label_count, 0);

    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const auto& labels = group_layout.groups[label];
        for (size_t bit_idx = 0; bit_idx < labels.size(); ++bit_idx) {
            const uint32_t original_label = labels[bit_idx];
            const uint64_t filter_bit = static_cast<uint64_t>(1) << bit_idx;
            metadata.label_filter_bit[original_label] = filter_bit;
        }
    }

    metadata.original_label_cardinality.assign(metadata.original_label_count, 0);
    for (uint32_t group_id = 0; group_id < metadata.label_count; ++group_id) {
        const auto& labels = group_layout.groups[group_id];
        const auto& masks = posting_data.filter_masks_by_group[group_id];
        if (labels.empty()) {
            if (!masks.empty()) {
                fail_graph("group posting data has rows for an empty group");
            }
            continue;
        }
        if (labels.size() == 1) {
            if (masks.size() > std::numeric_limits<uint32_t>::max()) {
                fail_graph("original-label cardinality exceeds uint32 range");
            }
            metadata.original_label_cardinality[labels.front()] =
                static_cast<uint32_t>(masks.size());
            continue;
        }

        const uint64_t allowed_mask = labels.size() == 64
            ? std::numeric_limits<uint64_t>::max()
            : ((1ULL << labels.size()) - 1ULL);
        for (uint64_t mask : masks) {
            if (mask == 0 || (mask & ~allowed_mask) != 0) {
                fail_graph("group posting data has an invalid original-label filter mask");
            }
            while (mask != 0) {
                const uint32_t bit_idx = static_cast<uint32_t>(__builtin_ctzll(mask));
                const uint32_t original_label = labels[bit_idx];
                uint32_t& cardinality = metadata.original_label_cardinality[original_label];
                if (cardinality == std::numeric_limits<uint32_t>::max()) {
                    fail_graph("original-label cardinality exceeds uint32 range");
                }
                ++cardinality;
                mask &= mask - 1;
            }
        }
    }

    uint64_t total_rows = 0;
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        metadata.label_offset[label] = static_cast<uint32_t>(total_rows);
        if (metadata.cat_freq[label] > static_cast<uint32_t>(specificity_threshold)) {
            metadata.label_size[label] = static_cast<uint32_t>(posting_data.sample_ids_by_group[label].size());
            total_rows += metadata.label_size[label];
        }
    }
    if (total_rows > std::numeric_limits<uint32_t>::max()) {
        fail_graph("row id exceeds uint32 range");
    }
    metadata.total_rows = static_cast<uint32_t>(total_rows);
    metadata.row_to_sample.resize(metadata.total_rows);
    metadata.row_filter_mask.resize(metadata.total_rows, 0);

    uint32_t cursor = 0;
    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        if (metadata.label_size[label] == 0) {
            continue;
        }
        const auto& samples = posting_data.sample_ids_by_group[label];
        const auto& masks = posting_data.filter_masks_by_group[label];
        for (size_t i = 0; i < samples.size(); ++i) {
            metadata.row_to_sample[cursor] = samples[i];
            metadata.row_filter_mask[cursor] = masks[i];
            ++cursor;
        }
    }

    return metadata;
}

LowfreqPostingIndex build_lowfreq_posting_index_from_posting_data(const GroupPostingData& posting_data,
                                                                  uint32_t group_count,
                                                                  int specificity_threshold)
{
    LowfreqPostingIndex index;
    index.group_count = group_count;
    index.label_size.assign(index.group_count, 0);
    index.label_offset.assign(index.group_count, 0);

    uint32_t cursor = 0;
    for (uint32_t group_id = 0; group_id < index.group_count; ++group_id) {
        index.label_offset[group_id] = cursor;
        const uint32_t group_size = static_cast<uint32_t>(posting_data.sample_ids_by_group[group_id].size());
        if (group_size <= static_cast<uint32_t>(specificity_threshold)) {
            index.label_size[group_id] = group_size;
            cursor += group_size;
        }
    }

    index.total_rows = cursor;
    index.sample_ids.resize(index.total_rows);
    index.row_filter_mask.resize(index.total_rows);

    cursor = 0;
    for (uint32_t group_id = 0; group_id < index.group_count; ++group_id) {
        const uint32_t group_size = index.label_size[group_id];
        if (group_size == 0) {
            continue;
        }
        const auto& samples = posting_data.sample_ids_by_group[group_id];
        const auto& masks = posting_data.filter_masks_by_group[group_id];
        for (uint32_t i = 0; i < group_size; ++i) {
            index.sample_ids[cursor] = samples[i];
            index.row_filter_mask[cursor] = masks[i];
            ++cursor;
        }
    }
    return index;
}

GraphMetadata rebuild_graph_metadata(const SparseLabelMatrix& data_labels, int specificity_threshold)
{
    return rebuild_graph_metadata(data_labels,
                                    specificity_threshold,
                                    build_identity_group_layout(data_labels.cols));
}

GraphMetadata rebuild_graph_metadata(const SparseLabelMatrix& data_labels,
                                         int specificity_threshold,
                                         const GroupLayout& group_layout)
{
    const GroupPostingData posting_data = build_group_posting_data(data_labels, group_layout);
    return build_graph_metadata_from_posting_data(data_labels,
                                                   specificity_threshold,
                                                   group_layout,
                                                   posting_data);
}

LowfreqPostingIndex build_lowfreq_posting_index(const SparseLabelMatrix& data_labels,
                                                const GroupLayout& group_layout,
                                                int specificity_threshold)
{
    const GroupPostingData posting_data = build_group_posting_data(data_labels, group_layout);
    return build_lowfreq_posting_index_from_posting_data(posting_data,
                                                        static_cast<uint32_t>(group_layout.groups.size()),
                                                        specificity_threshold);
}

GraphBuildIndexes build_graph_build_indexes(const SparseLabelMatrix& data_labels,
                                                           int specificity_threshold,
                                                           const GroupLayout& group_layout)
{
    const auto total_start = LoadClock::now();
    std::printf("[graph-build-prep] begin rows=%u labels=%u groups=%zu threshold=%d\n",
                data_labels.rows,
                data_labels.cols,
                group_layout.groups.size(),
                specificity_threshold);
    std::fflush(stdout);

    const auto posting_start = LoadClock::now();
    const GroupPostingData posting_data = build_group_posting_data(data_labels, group_layout);
    log_graph_build_stage("posting_ready", posting_start);

    const auto metadata_start = LoadClock::now();
    GraphBuildIndexes out;
    out.metadata = build_graph_metadata_from_posting_data(data_labels,
                                                           specificity_threshold,
                                                           group_layout,
                                                           posting_data);
    log_graph_build_stage("metadata_from_posting", metadata_start);

    const auto lowfreq_start = LoadClock::now();
    out.lowfreq_index = build_lowfreq_posting_index_from_posting_data(
        posting_data,
        static_cast<uint32_t>(group_layout.groups.size()),
        specificity_threshold);
    log_graph_build_stage("lowfreq_from_posting", lowfreq_start);

    std::printf("[graph-build-prep] done total_rows=%u lowfreq_rows=%u\n",
                out.metadata.total_rows,
                out.lowfreq_index.total_rows);
    std::fflush(stdout);
    log_graph_build_stage("total", total_start);
    return out;
}

template <typename Matrix>
std::vector<uint32_t> build_seed_local_ids_typed(const Matrix& base_vectors,
                                                 const GraphMetadata& metadata,
                                                 int dim)
{
    std::vector<uint32_t> seed_local_id_by_label(metadata.label_count, UINT32_MAX);
    std::vector<double> centroid(static_cast<size_t>(dim), 0.0);

    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        if (label_size == 0) {
            continue;
        }

        std::fill(centroid.begin(), centroid.end(), 0.0);
        const uint32_t begin = metadata.label_offset[label];
        for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
            const uint32_t global_row = begin + local_row;
            const auto* vec = base_vectors.row_ptr(metadata.row_to_sample[global_row]);
            for (int d = 0; d < dim; ++d) {
                centroid[static_cast<size_t>(d)] += static_cast<double>(vec[d]);
            }
        }
        for (int d = 0; d < dim; ++d) {
            centroid[static_cast<size_t>(d)] /= static_cast<double>(label_size);
        }

        double best_distance = std::numeric_limits<double>::max();
        uint32_t best_local_row = UINT32_MAX;
        for (uint32_t local_row = 0; local_row < label_size; ++local_row) {
            const uint32_t global_row = begin + local_row;
            const auto* vec = base_vectors.row_ptr(metadata.row_to_sample[global_row]);
            double dist = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double diff = static_cast<double>(vec[d]) - centroid[static_cast<size_t>(d)];
                dist += diff * diff;
            }
            if (dist < best_distance) {
                best_distance = dist;
                best_local_row = local_row;
            }
        }
        seed_local_id_by_label[label] = best_local_row;
    }

    return seed_local_id_by_label;
}

std::vector<uint32_t> build_seed_local_ids(const DenseUint8Matrix& base_vectors,
                                           const GraphMetadata& metadata,
                                           int dim)
{
    return build_seed_local_ids_typed(base_vectors, metadata, dim);
}

std::vector<uint32_t> build_seed_local_ids(const DenseFloatMatrix& base_vectors,
                                           const GraphMetadata& metadata,
                                           int dim)
{
    return build_seed_local_ids_typed(base_vectors, metadata, dim);
}

template <typename Matrix>
std::vector<uint32_t> build_seeds_by_original_label_typed(const Matrix& base_vectors,
                                                          const GraphMetadata& metadata,
                                                          const GroupLayout& group_layout,
                                                          int dim)
{
    std::vector<uint32_t> seed_by_original_label(metadata.original_label_count, UINT32_MAX);
    std::vector<double> centroid(static_cast<size_t>(dim), 0.0);

    for (uint32_t original_label = 0; original_label < metadata.original_label_count; ++original_label) {
        if (!has_matching_highfreq_rows(metadata, group_layout, original_label)) {
            continue;
        }

        const uint32_t group_id = group_layout.label_to_group[original_label];
        const uint32_t group_size = metadata.label_size[group_id];
        const uint64_t filter_bit = metadata.label_filter_bit[original_label];
        const uint32_t row_begin = metadata.label_offset[group_id];

        std::vector<uint32_t> matching_local_rows;
        for (uint32_t local_row = 0; local_row < group_size; ++local_row) {
            const uint32_t global_row = row_begin + local_row;
            if (metadata.implicit_all_ones_row_filter_mask ||
                (metadata.row_filter_mask[global_row] & filter_bit) != 0) {
                matching_local_rows.push_back(local_row);
            }
        }

        if (matching_local_rows.empty()) {
            continue;
        }

        std::fill(centroid.begin(), centroid.end(), 0.0);
        for (uint32_t local_row : matching_local_rows) {
            const uint32_t global_row = row_begin + local_row;
            const auto* vec = base_vectors.row_ptr(metadata.row_to_sample[global_row]);
            for (int d = 0; d < dim; ++d) {
                centroid[static_cast<size_t>(d)] += static_cast<double>(vec[d]);
            }
        }
        for (int d = 0; d < dim; ++d) {
            centroid[static_cast<size_t>(d)] /= static_cast<double>(matching_local_rows.size());
        }

        double best_distance = std::numeric_limits<double>::max();
        uint32_t best_local_row = UINT32_MAX;
        for (uint32_t local_row : matching_local_rows) {
            const uint32_t global_row = row_begin + local_row;
            const auto* vec = base_vectors.row_ptr(metadata.row_to_sample[global_row]);
            double dist = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double diff = static_cast<double>(vec[d]) - centroid[static_cast<size_t>(d)];
                dist += diff * diff;
            }
            if (dist < best_distance) {
                best_distance = dist;
                best_local_row = local_row;
            }
        }
        seed_by_original_label[original_label] = best_local_row;
    }

    return seed_by_original_label;
}

std::vector<uint32_t> build_seeds_by_original_label(const DenseUint8Matrix& base_vectors,
                                                    const GraphMetadata& metadata,
                                                    const GroupLayout& group_layout,
                                                    int dim)
{
    return build_seeds_by_original_label_typed(base_vectors, metadata, group_layout, dim);
}

std::vector<uint32_t> build_seeds_by_original_label(const DenseFloatMatrix& base_vectors,
                                                    const GraphMetadata& metadata,
                                                    const GroupLayout& group_layout,
                                                    int dim)
{
    return build_seeds_by_original_label_typed(base_vectors, metadata, group_layout, dim);
}

bool load_seed_sidecar(const std::string& path,
                       const GraphMetadata& metadata,
                       uint64_t num_points,
                       uint64_t graph_degree,
                       int specificity_threshold,
                       std::vector<uint32_t>& seed_local_id_by_label)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }

    GraphSeedFileHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        return false;
    }

    const bool compatible =
        header.version == kGraphSeedVersion &&
        header.label_count == metadata.label_count &&
        header.num_points == num_points &&
        header.graph_rows == metadata.total_rows &&
        header.graph_degree == graph_degree &&
        header.specificity_threshold == static_cast<uint64_t>(specificity_threshold);
    if (!compatible) {
        return false;
    }

    seed_local_id_by_label.resize(metadata.label_count);
    input.read(reinterpret_cast<char*>(seed_local_id_by_label.data()),
               seed_local_id_by_label.size() * sizeof(uint32_t));
    if (!input) {
        return false;
    }

    for (uint32_t label = 0; label < metadata.label_count; ++label) {
        const uint32_t label_size = metadata.label_size[label];
        const uint32_t seed = seed_local_id_by_label[label];
        if (label_size == 0) {
            if (seed != UINT32_MAX) {
                return false;
            }
            continue;
        }
        if (seed == UINT32_MAX || seed >= label_size) {
            return false;
        }
    }
    return true;
}

void save_seed_sidecar(const std::string& path,
                       const GraphMetadata& metadata,
                       uint64_t num_points,
                       uint64_t graph_degree,
                       int specificity_threshold,
                       const std::vector<uint32_t>& seed_local_id_by_label)
{
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        fail_graph("unable to write seed sidecar " + path);
    }

    GraphSeedFileHeader header;
    header.label_count = metadata.label_count;
    header.num_points = num_points;
    header.graph_rows = metadata.total_rows;
    header.graph_degree = graph_degree;
    header.specificity_threshold = static_cast<uint64_t>(specificity_threshold);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(seed_local_id_by_label.data()),
                 seed_local_id_by_label.size() * sizeof(uint32_t));
    if (!output) {
        fail_graph("failed to write seed sidecar " + path);
    }
}

template <typename T>
bool read_vector_payload(std::ifstream& input,
                         const std::string& path,
                         const char* field_name,
                         size_t count,
                         std::vector<T>& out)
{
    out.resize(count);
    if (count == 0) {
        return true;
    }
    input.read(reinterpret_cast<char*>(out.data()),
               static_cast<std::streamsize>(out.size() * sizeof(T)));
    if (!input) {
        std::fprintf(stderr,
                     "[runtime] metadata sidecar '%s' ended while reading %s\n",
                     path.c_str(),
                     field_name);
        return false;
    }
    return true;
}

template <typename T>
void write_vector_payload(std::ofstream& output, const std::vector<T>& values)
{
    if (!values.empty()) {
        output.write(reinterpret_cast<const char*>(values.data()),
                     static_cast<std::streamsize>(values.size() * sizeof(T)));
    }
}

bool validate_metadata_sidecar_vectors(const GraphMetadata& metadata,
                                       uint64_t num_points,
                                       const GroupLayout& group_layout,
                                       const std::string& path,
                                       bool implicit_all_ones_row_filter_mask)
{
    const auto fail = [&](const std::string& reason) {
        std::fprintf(stderr,
                     "[runtime] metadata sidecar '%s' incompatible: %s\n",
                     path.c_str(),
                     reason.c_str());
        return false;
    };

    if (metadata.cat_freq.size() != metadata.label_count ||
        metadata.group_posting_size.size() != metadata.label_count ||
        metadata.label_size.size() != metadata.label_count ||
        metadata.label_offset.size() != metadata.label_count) {
        return fail("group-dimension vector size mismatch");
    }
    if (metadata.label_filter_bit.size() != metadata.original_label_count ||
        (!metadata.original_label_cardinality.empty() &&
         metadata.original_label_cardinality.size() != metadata.original_label_count)) {
        return fail("original-label-dimension vector size mismatch");
    }
    if (metadata.row_to_sample.size() != metadata.total_rows ||
        (!implicit_all_ones_row_filter_mask &&
         metadata.row_filter_mask.size() != metadata.total_rows) ||
        (implicit_all_ones_row_filter_mask && !metadata.row_filter_mask.empty())) {
        return fail("row-dimension vector size mismatch");
    }

    uint64_t expected_total_rows = 0;
    for (uint32_t group_id = 0; group_id < metadata.label_count; ++group_id) {
        if (metadata.cat_freq[group_id] != metadata.group_posting_size[group_id]) {
            return fail("cat_freq/group_posting_size mismatch");
        }
        if (metadata.label_offset[group_id] != expected_total_rows) {
            return fail("label_offset is not contiguous");
        }
        expected_total_rows += metadata.label_size[group_id];
        if (metadata.label_size[group_id] > 0 &&
            metadata.cat_freq[group_id] <= static_cast<uint32_t>(metadata.specificity_threshold)) {
            return fail("high-frequency label_size conflicts with specificity_threshold");
        }
    }
    if (expected_total_rows != metadata.total_rows) {
        return fail("label_size sum does not match total_rows");
    }
    for (uint32_t row = 0; row < metadata.total_rows; ++row) {
        if (metadata.row_to_sample[row] >= num_points) {
            return fail("row_to_sample exceeds num_points");
        }
    }
    if (group_layout.groups.size() != metadata.label_count ||
        group_layout.label_to_group.size() != metadata.original_label_count) {
        return fail("group-layout dimension mismatch");
    }
    for (uint32_t group_id = 0; group_id < metadata.label_count; ++group_id) {
        const auto& labels = group_layout.groups[group_id];
        if (labels.size() > 64) {
            return fail("group size exceeds row_filter_mask bit capacity");
        }
        for (size_t bit_idx = 0; bit_idx < labels.size(); ++bit_idx) {
            const uint32_t original_label = labels[bit_idx];
            if (original_label >= metadata.original_label_count) {
                return fail("group layout contains original label outside sidecar label count");
            }
            if (group_layout.label_to_group[original_label] != group_id) {
                return fail("group layout label_to_group is inconsistent with group membership");
            }
            const uint64_t expected_bit = 1ULL << bit_idx;
            if (metadata.label_filter_bit[original_label] != expected_bit) {
                return fail("label_filter_bit does not match group-local label bit");
            }
        }
    }
    return true;
}

bool load_metadata_sidecar(const std::string& path,
                           uint64_t num_points,
                           uint64_t row_limit,
                           uint64_t graph_rows,
                           int specificity_threshold,
                           const SpmatHeaderInfo& label_header,
                           const GroupLayout& group_layout,
                           GraphMetadata& metadata,
                           bool implicit_all_ones_row_filter_mask)
{
    if (path.empty()) {
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }

    GraphMetadataSidecarHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        return false;
    }

    const uint64_t checksum = group_layout_checksum(group_layout);
    const bool compatible =
        header.version == kGraphMetadataSidecarVersion &&
        header.label_count == group_layout.groups.size() &&
        header.group_count == group_layout.groups.size() &&
        header.original_label_count == label_header.cols &&
        header.original_label_count == group_layout.label_to_group.size() &&
        header.num_points == num_points &&
        header.row_limit == row_limit &&
        header.graph_rows == graph_rows &&
        header.total_rows == graph_rows &&
        header.specificity_threshold == static_cast<uint64_t>(specificity_threshold) &&
        header.group_layout_checksum == checksum;
    if (!compatible) {
        std::fprintf(stderr,
                     "[runtime] metadata sidecar '%s' incompatible; rebuilding metadata\n",
                     path.c_str());
        return false;
    }
    if (header.label_count > std::numeric_limits<uint32_t>::max() ||
        header.original_label_count > std::numeric_limits<uint32_t>::max() ||
        header.total_rows > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    GraphMetadata candidate;
    candidate.label_count = static_cast<uint32_t>(header.label_count);
    candidate.original_label_count = static_cast<uint32_t>(header.original_label_count);
    candidate.total_rows = static_cast<uint32_t>(header.total_rows);
    candidate.specificity_threshold = static_cast<int>(header.specificity_threshold);
    candidate.implicit_all_ones_row_filter_mask = implicit_all_ones_row_filter_mask;

    if (!read_vector_payload(input, path, "cat_freq", candidate.label_count, candidate.cat_freq) ||
        !read_vector_payload(input, path, "group_posting_size", candidate.label_count, candidate.group_posting_size) ||
        !read_vector_payload(input, path, "label_size", candidate.label_count, candidate.label_size) ||
        !read_vector_payload(input, path, "label_offset", candidate.label_count, candidate.label_offset) ||
        !read_vector_payload(input, path, "row_to_sample", candidate.total_rows, candidate.row_to_sample) ||
        !(implicit_all_ones_row_filter_mask
              ? (input.seekg(static_cast<std::streamoff>(candidate.total_rows) *
                                 static_cast<std::streamoff>(sizeof(uint64_t)),
                             std::ios::cur),
                 static_cast<bool>(input))
              : read_vector_payload(input, path, "row_filter_mask", candidate.total_rows,
                                    candidate.row_filter_mask)) ||
        !read_vector_payload(input, path, "label_filter_bit", candidate.original_label_count, candidate.label_filter_bit)) {
        return false;
    }

    if (!validate_metadata_sidecar_vectors(candidate,
                                           num_points,
                                           group_layout,
                                           path,
                                           implicit_all_ones_row_filter_mask)) {
        return false;
    }
    metadata = std::move(candidate);
    std::fprintf(stdout,
                 "[runtime] loaded metadata sidecar %s (groups=%u labels=%u rows=%u)\n",
                 path.c_str(),
                 metadata.label_count,
                 metadata.original_label_count,
                 metadata.total_rows);
    return true;
}

void save_metadata_sidecar(const std::string& path,
                           const GraphMetadata& metadata,
                           uint64_t num_points,
                           uint64_t row_limit,
                           uint64_t graph_rows,
                           uint64_t graph_degree,
                           int specificity_threshold,
                           const GroupLayout& group_layout)
{
    if (path.empty()) {
        return;
    }
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        fail_graph("unable to write metadata sidecar " + path);
    }

    GraphMetadataSidecarHeader header;
    header.label_count = metadata.label_count;
    header.group_count = metadata.label_count;
    header.original_label_count = metadata.original_label_count;
    header.num_points = num_points;
    header.row_limit = row_limit;
    header.graph_rows = graph_rows;
    header.total_rows = metadata.total_rows;
    header.graph_degree = graph_degree;
    header.specificity_threshold = static_cast<uint64_t>(specificity_threshold);
    header.group_layout_checksum = group_layout_checksum(group_layout);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    write_vector_payload(output, metadata.cat_freq);
    write_vector_payload(output, metadata.group_posting_size);
    write_vector_payload(output, metadata.label_size);
    write_vector_payload(output, metadata.label_offset);
    write_vector_payload(output, metadata.row_to_sample);
    write_vector_payload(output, metadata.row_filter_mask);
    write_vector_payload(output, metadata.label_filter_bit);
    if (!output) {
        fail_graph("failed to write metadata sidecar " + path);
    }
    std::fprintf(stdout,
                 "[runtime] wrote metadata sidecar %s (groups=%u labels=%u rows=%u)\n",
                 path.c_str(),
                 metadata.label_count,
                 metadata.original_label_count,
                 metadata.total_rows);
}

bool should_replace_route_choice(const std::string& routing_mode,
                                 uint32_t current_posting,
                                 uint32_t best_posting)
{
    if (routing_mode == "greedy_min_posting") {
        return current_posting < best_posting;
    }
    if (routing_mode == "greedy_max_posting") {
        return current_posting > best_posting;
    }
    fail_graph("unsupported routing_mode: " + routing_mode);
}

bool query_range_active(uint8_t mode)
{
    return mode != 0;
}

bool query_has_any_active_range(const QueryRangeTable& ranges, uint32_t query_idx)
{
    if (query_idx >= ranges.rows) {
        fail_graph("query range row index out of range");
    }
    for (uint32_t range_idx = 0; range_idx < ranges.range_count; ++range_idx) {
        if (query_range_active(ranges.modes[ranges.flat_index(query_idx, range_idx)])) {
            return true;
        }
    }
    return false;
}

bool interval_contains_point(float lower, float upper, float point)
{
    constexpr float kIntervalEps = 1e-6f;
    return lower <= point + kIntervalEps && upper > point + kIntervalEps;
}

bool intervals_overlap(float left_low, float left_high, float right_low, float right_high)
{
    constexpr float kIntervalEps = 1e-6f;
    return left_high > right_low + kIntervalEps && left_low < right_high - kIntervalEps;
}

bool range_route_label_usable(const GraphRuntime& runtime, uint32_t label, uint32_t range_idx)
{
    if (label >= runtime.label_intervals.has_interval.size()) {
        return false;
    }
    if (!runtime.label_intervals.has_interval[label]) {
        return false;
    }
    if (label >= runtime.label_intervals.range_indices.size() ||
        runtime.label_intervals.range_indices[label] != range_idx) {
        return false;
    }
    if (label >= runtime.group_layout.label_to_group.size()) {
        return false;
    }
    const uint32_t group = runtime.group_layout.label_to_group[label];
    if (group == UINT32_MAX || group >= runtime.metadata.label_count) {
        return false;
    }
    if (runtime.metadata.label_filter_bit[label] == 0) {
        return false;
    }
    const bool is_highfreq = runtime.metadata.label_size[group] > 0;
    const bool is_lowfreq = !is_highfreq &&
                            group < runtime.lowfreq_index.label_size.size() &&
                            runtime.lowfreq_index.label_size[group] > 0;
    return is_highfreq || is_lowfreq;
}

enum class RouteResidualMode : uint8_t {
    RemainingConjunction = 0,
    None = 1,
};

struct RouteLabelCandidate {
    uint32_t original_label = UINT32_MAX;
    uint32_t group = UINT32_MAX;
    uint32_t posting_size = 0;
    uint64_t filter_bit = 0;
    bool is_lowfreq = false;
    uint32_t original_position = 0;
    uint64_t physical_cost = 0;
    uint32_t exact_lowfreq_rows = 0;
    uint64_t highfreq_cost_proxy = 0;
    uint32_t exact_label_cardinality = 0;
    QueryLabelSourceTier source_tier = QueryLabelSourceTier::Exact;
    uint32_t relevance_rank = 0;
    double predicted_latency_us = 0.0;
    double estimated_marginal_coverage = 0.0;
    double quality = 0.0;
    double utility = 0.0;
};

struct OrRouteCandidateTraceRow {
    uint32_t original_query_id = 0;
    uint32_t routed_route_id = 0;
    uint32_t label = 0;
    QueryLabelSourceTier tier = QueryLabelSourceTier::Exact;
    uint32_t rank = 0;
    uint32_t exact_label_cardinality = 0;
    bool is_lowfreq = false;
    uint32_t lowfreq_rows = 0;
    uint32_t highfreq_graph_rows = 0;
    uint64_t highfreq_degree = 0;
    uint64_t highfreq_work = 0;
};

void write_or_route_calibration_trace(
    const std::string& path,
    const std::string& run_id,
    const std::vector<OrRouteCandidateTraceRow>& rows)
{
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) {
        fail_graph("unable to write OR route calibration trace " + path);
    }
    for (const OrRouteCandidateTraceRow& row : rows) {
        const json encoded = {
            {"schema", "flashanns.or_route_candidate_trace"},
            {"version", 1},
            {"run_id", run_id},
            {"original_query_id", row.original_query_id},
            {"routed_route_id", row.routed_route_id},
            {"label", row.label},
            {"tier", query_label_source_tier_name(row.tier)},
            {"rank", row.rank},
            {"exact_label_cardinality", row.exact_label_cardinality},
            {"route_type", row.is_lowfreq ? "low" : "high"},
            {"lowfreq_rows", row.lowfreq_rows},
            {"highfreq_graph_rows", row.highfreq_graph_rows},
            {"highfreq_degree", row.highfreq_degree},
            {"highfreq_work", row.highfreq_work},
        };
        output << encoded.dump() << '\n';
    }
    output.close();
    if (!output) {
        fail_graph("failed to write OR route calibration trace " + path);
    }
}

uint64_t saturating_add_u64(uint64_t left, uint64_t right)
{
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left + right;
}

void saturating_increment_u64(uint64_t& value)
{
    value = saturating_add_u64(value, 1);
}

uint64_t saturating_multiply_u64(uint64_t left, uint64_t right)
{
    if (left != 0 && right > std::numeric_limits<uint64_t>::max() / left) {
        return std::numeric_limits<uint64_t>::max();
    }
    return left * right;
}

uint64_t or_route_hash_global_sample_id(uint64_t sample_id, uint64_t seed)
{
    uint64_t value = sample_id ^ seed;
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

uint64_t runtime_num_points_for_sketch(const GraphRuntime& runtime)
{
    if (runtime.base_vector_metadata.rows > 0) {
        return runtime.base_vector_metadata.rows;
    }
    if (runtime.base_vectors.rows > 0) {
        return runtime.base_vectors.rows;
    }
    uint64_t max_sample_plus_one = 0;
    for (uint32_t sample_id : runtime.metadata.row_to_sample) {
        max_sample_plus_one = std::max(max_sample_plus_one,
                                       static_cast<uint64_t>(sample_id) + 1);
    }
    for (uint32_t sample_id : runtime.lowfreq_index.sample_ids) {
        max_sample_plus_one = std::max(max_sample_plus_one,
                                       static_cast<uint64_t>(sample_id) + 1);
    }
    return max_sample_plus_one;
}

std::vector<uint64_t> merge_or_route_bottom_k(const std::vector<uint64_t>& left,
                                              const std::vector<uint64_t>& right,
                                              size_t sketch_k)
{
    std::vector<uint64_t> merged;
    merged.reserve(std::min(sketch_k, left.size() + right.size()));
    size_t left_idx = 0;
    size_t right_idx = 0;
    while (merged.size() < sketch_k &&
           (left_idx < left.size() || right_idx < right.size())) {
        uint64_t value = 0;
        if (right_idx >= right.size() ||
            (left_idx < left.size() && left[left_idx] < right[right_idx])) {
            value = left[left_idx++];
        } else if (left_idx >= left.size() || right[right_idx] < left[left_idx]) {
            value = right[right_idx++];
        } else {
            value = left[left_idx];
            ++left_idx;
            ++right_idx;
        }
        if (merged.empty() || merged.back() != value) {
            merged.push_back(value);
        }
    }
    return merged;
}

double estimate_or_route_kmv_cardinality(const std::vector<uint64_t>& sketch,
                                         size_t sketch_k,
                                         uint64_t num_points)
{
    if (sketch.size() < sketch_k) {
        return static_cast<double>(sketch.size());
    }
    if (sketch.empty() || sketch_k < 2) {
        return 0.0;
    }
    const long double denominator = static_cast<long double>(sketch.back()) + 1.0L;
    const long double hash_domain = std::ldexp(1.0L, 64);
    long double estimate = static_cast<long double>(sketch_k - 1) * hash_domain / denominator;
    estimate = std::max<long double>(estimate, static_cast<long double>(sketch.size()));
    estimate = std::min<long double>(estimate, static_cast<long double>(num_points));
    return static_cast<double>(estimate);
}

class ActiveOrRouteSketches {
public:
    ActiveOrRouteSketches(const GraphRuntime& runtime,
                          const OrRoutePlannerRuntimeSelection& selection,
                          OrRoutePlannerProfile& profile)
        : runtime_(runtime), selection_(selection), profile_(profile),
          num_points_(runtime_num_points_for_sketch(runtime))
    {
        const OrRouteAccuracyCostModel& model = selection_.model;
        if (model.token_count != runtime.metadata.original_label_count ||
            model.group_count != runtime.group_layout.groups.size() ||
            model.group_layout_checksum != group_layout_checksum(runtime.group_layout)) {
            fail_graph("calibrated model is incompatible with runtime label/group layout");
        }
        if (num_points_ == 0 || model.num_points != num_points_ ||
            model.max_row_id_exclusive != num_points_) {
            fail_graph("calibrated model point count is incompatible with runtime data");
        }
        profile_.calibration_id = model.calibration_id;
        if (selection_.config.sketch_sidecar_path.empty()) {
            if (!selection_.config.allow_runtime_sketch_fallback) {
                fail_graph("marginal planner has no sketch sidecar");
            }
            if (num_points_ > selection_.config.runtime_sketch_fallback_max_rows) {
                fail_graph(
                    "runtime sketch fallback exceeds its small-fixture row limit; build the offline sidecar");
            }
            return;
        }
        try {
            open_sidecar();
        } catch (...) {
            if (fd_ >= 0) {
                close(fd_);
                fd_ = -1;
            }
            throw;
        }
    }

    ~ActiveOrRouteSketches()
    {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    const std::vector<uint64_t>& get(uint32_t original_label,
                                     uint32_t exact_label_cardinality)
    {
        auto existing = sketches_.find(original_label);
        if (existing != sketches_.end()) {
            return existing->second;
        }
        std::vector<uint64_t> sketch = fd_ >= 0
            ? load_sidecar_sketch(original_label, exact_label_cardinality)
            : build_runtime_sketch(original_label, exact_label_cardinality);
        profile_.sketch_hashes = saturating_add_u64(profile_.sketch_hashes, sketch.size());
        profile_.sketch_bytes = saturating_add_u64(
            profile_.sketch_bytes, sketch.size() * sizeof(uint64_t));
        saturating_increment_u64(profile_.sketch_labels);
        auto inserted = sketches_.emplace(original_label, std::move(sketch));
        return inserted.first->second;
    }

    uint64_t num_points() const { return num_points_; }

private:
    void open_sidecar()
    {
        const std::string& path = selection_.config.sketch_sidecar_path;
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            fail_graph("unable to open OR route sketch sidecar " + path);
        }
        read_exact_at(fd_, &header_, sizeof(header_), 0, "OR route sketch header");
        profile_.sketch_sidecar_bytes_read = saturating_add_u64(
            profile_.sketch_sidecar_bytes_read, sizeof(header_));
        constexpr char kMagic[16] = {'F','A','N','N','S','K','M','V','S','K','E','T','C','H','\0','\0'};
        constexpr char kProvenance[] = "tokpost_v1";
        if (!std::equal(std::begin(header_.magic), std::end(header_.magic), std::begin(kMagic)) ||
            header_.version != kOrRouteSketchSidecarVersion ||
            header_.header_bytes != sizeof(header_) ||
            header_.endian != 0x01020304 || header_.flags != 1 ||
            header_.checksum_kind != 1 ||
            std::strncmp(header_.provenance, kProvenance, sizeof(kProvenance) - 1) != 0) {
            fail_graph("invalid OR route sketch sidecar header in " + path);
        }
        if (header_.sketch_k < kOrRouteSketchMinK ||
            header_.sketch_k > kOrRouteSketchMaxK ||
            header_.source_posting_version != 1 ||
            header_.source_posting_token_count != header_.token_count ||
            header_.source_posting_max_row_id_exclusive != header_.max_row_id_exclusive ||
            header_.num_points != header_.max_row_id_exclusive) {
            fail_graph("unsupported OR route sketch sidecar metadata in " + path);
        }
        const bool source_checksum_nonzero = std::any_of(
            std::begin(header_.source_posting_sha256),
            std::end(header_.source_posting_sha256),
            [](unsigned char value) { return value != 0; });
        const bool payload_checksum_nonzero = std::any_of(
            std::begin(header_.payload_sha256),
            std::end(header_.payload_sha256),
            [](unsigned char value) { return value != 0; });
        const uint64_t source_offsets_count = header_.source_posting_token_count + 1;
        if (!source_checksum_nonzero || !payload_checksum_nonzero ||
            source_offsets_count < header_.source_posting_token_count ||
            source_offsets_count >
                (std::numeric_limits<uint64_t>::max() - 64) / sizeof(uint64_t) ||
            header_.source_posting_count >
                (std::numeric_limits<uint64_t>::max() - 64 -
                 source_offsets_count * sizeof(uint64_t)) / sizeof(uint32_t) ||
            header_.source_posting_file_bytes !=
                64 + source_offsets_count * sizeof(uint64_t) +
                    header_.source_posting_count * sizeof(uint32_t)) {
            fail_graph("invalid OR route sketch source posting provenance in " + path);
        }
        const OrRouteAccuracyCostModel& model = selection_.model;
        if (header_.token_count != model.token_count ||
            header_.num_points != model.num_points ||
            header_.max_row_id_exclusive != model.max_row_id_exclusive ||
            header_.sketch_k != model.sketch_k || header_.hash_seed != model.hash_seed ||
            header_.group_count != model.group_count ||
            header_.group_layout_checksum != model.group_layout_checksum ||
            !std::equal(std::begin(header_.payload_sha256),
                        std::end(header_.payload_sha256),
                        model.sketch_payload_sha256.begin()) ||
            !std::equal(std::begin(header_.source_posting_sha256),
                        std::end(header_.source_posting_sha256),
                        model.source_posting_sha256.begin())) {
            fail_graph("OR route sketch sidecar does not match the calibrated model");
        }
        const uint64_t offsets_count = header_.token_count + 1;
        if (offsets_count < header_.token_count ||
            offsets_count > (std::numeric_limits<uint64_t>::max() - sizeof(header_)) / 8 ||
            header_.hash_count > (std::numeric_limits<uint64_t>::max() - sizeof(header_) - offsets_count * 8) / 8) {
            fail_graph("OR route sketch sidecar size overflows uint64");
        }
        hash_payload_offset_ = sizeof(header_) + offsets_count * 8;
        const uint64_t expected_bytes = hash_payload_offset_ + header_.hash_count * 8;
        struct stat stat_buffer {};
        if (fstat(fd_, &stat_buffer) != 0 || stat_buffer.st_size < 0 ||
            static_cast<uint64_t>(stat_buffer.st_size) != expected_bytes) {
            fail_graph("OR route sketch sidecar has an incompatible exact file size");
        }
        uint64_t first_offset = 1;
        uint64_t last_offset = 0;
        read_exact_at(fd_, &first_offset, sizeof(first_offset), sizeof(header_),
                      "OR route sketch first offset");
        read_exact_at(fd_, &last_offset, sizeof(last_offset),
                      sizeof(header_) + header_.token_count * sizeof(uint64_t),
                      "OR route sketch last offset");
        profile_.sketch_sidecar_bytes_read = saturating_add_u64(
            profile_.sketch_sidecar_bytes_read, 2 * sizeof(uint64_t));
        if (first_offset != 0 || last_offset != header_.hash_count) {
            fail_graph("OR route sketch offsets do not span the hash payload");
        }
        if (selection_.config.verify_sketch_payload_checksum) {
            const std::array<uint8_t, 32> actual = sha256_file_payload(
                path, sizeof(header_), expected_bytes - sizeof(header_));
            if (!std::equal(actual.begin(), actual.end(), std::begin(header_.payload_sha256))) {
                fail_graph("OR route sketch payload checksum mismatch in " + path);
            }
            profile_.sketch_sidecar_bytes_read = saturating_add_u64(
                profile_.sketch_sidecar_bytes_read, expected_bytes - sizeof(header_));
        }
    }

    std::vector<uint64_t> load_sidecar_sketch(uint32_t label,
                                              uint32_t exact_cardinality)
    {
        if (label >= header_.token_count) {
            fail_graph("active OR route label is outside sketch sidecar token_count");
        }
        uint64_t offsets[2]{};
        read_exact_at(fd_, offsets, sizeof(offsets),
                      sizeof(header_) + static_cast<uint64_t>(label) * sizeof(uint64_t),
                      "active OR route sketch offsets");
        profile_.sketch_sidecar_bytes_read = saturating_add_u64(
            profile_.sketch_sidecar_bytes_read, sizeof(offsets));
        if (offsets[1] < offsets[0] || offsets[1] > header_.hash_count ||
            offsets[1] - offsets[0] > header_.sketch_k) {
            fail_graph("active OR route sketch has invalid offsets");
        }
        const uint64_t expected_length = std::min<uint64_t>(header_.sketch_k,
                                                            exact_cardinality);
        if (offsets[1] - offsets[0] != expected_length) {
            fail_graph("active OR route sketch length does not match exact label cardinality");
        }
        std::vector<uint64_t> sketch(static_cast<size_t>(expected_length));
        if (!sketch.empty()) {
            read_exact_at(fd_, sketch.data(), sketch.size() * sizeof(uint64_t),
                          hash_payload_offset_ + offsets[0] * sizeof(uint64_t),
                          "active OR route sketch hashes");
            profile_.sketch_sidecar_bytes_read = saturating_add_u64(
                profile_.sketch_sidecar_bytes_read,
                sketch.size() * sizeof(uint64_t));
            if (!std::is_sorted(sketch.begin(), sketch.end()) ||
                std::adjacent_find(sketch.begin(), sketch.end()) != sketch.end()) {
                fail_graph("active OR route sketch hashes are not strictly increasing");
            }
        }
        return sketch;
    }

    std::vector<uint64_t> build_runtime_sketch(uint32_t label,
                                               uint32_t exact_cardinality)
    {
        if (label >= runtime_.group_layout.label_to_group.size()) {
            fail_graph("runtime sketch label is outside the group layout");
        }
        const uint32_t group = runtime_.group_layout.label_to_group[label];
        if (group >= runtime_.group_layout.groups.size()) {
            fail_graph("runtime sketch group is outside the group layout");
        }
        const uint64_t filter_bit = runtime_.metadata.label_filter_bit[label];
        const uint32_t rows = runtime_.metadata.group_posting_size[group];
        const uint32_t* sample_ids = nullptr;
        const uint64_t* masks = nullptr;
        if (runtime_.metadata.label_size[group] > 0) {
            const uint64_t begin = runtime_.metadata.label_offset[group];
            if (begin + rows > runtime_.metadata.row_to_sample.size()) {
                fail_graph("runtime high-frequency sketch sample ids are truncated");
            }
            sample_ids = runtime_.metadata.row_to_sample.data() + begin;
            if (runtime_.group_layout.groups[group].size() > 1) {
                if (begin + rows > runtime_.metadata.row_filter_mask.size()) {
                    fail_graph("runtime high-frequency sketch masks are truncated");
                }
                masks = runtime_.metadata.row_filter_mask.data() + begin;
            }
        } else {
            if (group >= runtime_.lowfreq_index.label_offset.size() ||
                group >= runtime_.lowfreq_index.label_size.size() ||
                runtime_.lowfreq_index.label_size[group] != rows) {
                fail_graph("runtime low-frequency sketch posting metadata is inconsistent");
            }
            const uint64_t begin = runtime_.lowfreq_index.label_offset[group];
            if (begin + rows > runtime_.lowfreq_index.sample_ids.size()) {
                fail_graph("runtime low-frequency sketch sample ids are truncated");
            }
            sample_ids = runtime_.lowfreq_index.sample_ids.data() + begin;
            if (runtime_.group_layout.groups[group].size() > 1) {
                if (begin + rows > runtime_.lowfreq_index.row_filter_mask.size()) {
                    fail_graph("runtime low-frequency sketch masks are truncated");
                }
                masks = runtime_.lowfreq_index.row_filter_mask.data() + begin;
            }
        }
        profile_.sketch_build_rows = saturating_add_u64(profile_.sketch_build_rows, rows);
        std::vector<uint64_t> hashes;
        hashes.reserve(std::min<uint32_t>(selection_.config.sketch_k, exact_cardinality));
        for (uint32_t row = 0; row < rows; ++row) {
            if (masks != nullptr && (masks[row] & filter_bit) == 0) {
                continue;
            }
            if (sample_ids[row] >= num_points_) {
                fail_graph("runtime sketch sample id exceeds num_points");
            }
            hashes.push_back(or_route_hash_global_sample_id(
                sample_ids[row], selection_.config.sketch_hash_seed));
        }
        if (hashes.size() != exact_cardinality) {
            fail_graph("runtime sketch row count differs from exact label cardinality");
        }
        if (hashes.size() > selection_.config.sketch_k) {
            std::nth_element(hashes.begin(),
                             hashes.begin() + selection_.config.sketch_k,
                             hashes.end());
            hashes.resize(selection_.config.sketch_k);
        }
        std::sort(hashes.begin(), hashes.end());
        if (std::adjacent_find(hashes.begin(), hashes.end()) != hashes.end()) {
            fail_graph("runtime sketch hash collision or duplicate global sample id");
        }
        return hashes;
    }

    const GraphRuntime& runtime_;
    const OrRoutePlannerRuntimeSelection& selection_;
    OrRoutePlannerProfile& profile_;
    uint64_t num_points_ = 0;
    int fd_ = -1;
    OrRouteSketchSidecarHeader header_{};
    uint64_t hash_payload_offset_ = 0;
    std::unordered_map<uint32_t, std::vector<uint64_t>> sketches_;
};

void validate_query_occurrence_metadata(const SparseLabelMatrix& matrix,
                                        const std::string& context)
{
    const bool has_source_tiers = !matrix.source_tiers.empty();
    const bool has_relevance_ranks = !matrix.relevance_ranks.empty();
    if (!has_source_tiers && !has_relevance_ranks) {
        return;
    }
    const auto fail = [&](const std::string& message) {
        fail_graph(context + ": " + message);
    };
    if (!has_source_tiers || !has_relevance_ranks ||
        matrix.source_tiers.size() != matrix.indices.size() ||
        matrix.relevance_ranks.size() != matrix.indices.size()) {
        fail("source tiers and relevance ranks must both be CSR-aligned");
    }
    if (matrix.indptr.size() != static_cast<size_t>(matrix.rows) + 1 ||
        matrix.indptr.empty() || matrix.indptr.front() != 0 ||
        matrix.indptr.back() != matrix.indices.size()) {
        fail("occurrence metadata has an invalid CSR indptr");
    }
    if (!matrix.row_labels.empty() && matrix.row_labels.size() != matrix.rows) {
        fail("row label cache size does not match CSR rows");
    }

    for (uint32_t row = 0; row < matrix.rows; ++row) {
        const uint64_t begin = matrix.indptr[row];
        const uint64_t end = matrix.indptr[static_cast<size_t>(row) + 1];
        if (begin > end || end > matrix.indices.size()) {
            fail("occurrence metadata has an invalid CSR row span");
        }
        if (!matrix.row_labels.empty()) {
            const std::vector<uint32_t>& cached = matrix.row_labels[row];
            if (cached.size() != end - begin ||
                !std::equal(cached.begin(), cached.end(), matrix.indices.begin() + begin)) {
                fail("row label cache is not aligned with CSR indices");
            }
        }
    }

    for (size_t idx = 0; idx < matrix.indices.size(); ++idx) {
        const uint8_t source_tier = matrix.source_tiers[idx];
        const uint32_t relevance_rank = matrix.relevance_ranks[idx];
        if (source_tier > static_cast<uint8_t>(QueryLabelSourceTier::Fallback)) {
            fail("occurrence metadata has an invalid source tier");
        }
        if (source_tier == static_cast<uint8_t>(QueryLabelSourceTier::Fallback)) {
            if (relevance_rank == 0) {
                fail("fallback occurrence metadata requires a one-based relevance rank");
            }
        } else if (source_tier == static_cast<uint8_t>(QueryLabelSourceTier::Exact) &&
                   relevance_rank != 0) {
            fail("exact occurrence metadata requires relevance rank zero");
        }
    }
}

void ensure_exact_original_label_cardinality(GraphRuntime& runtime)
{
    GraphMetadata& metadata = runtime.metadata;
    const GroupLayout& group_layout = runtime.group_layout;
    if (metadata.original_label_count != group_layout.label_to_group.size() ||
        metadata.label_count != group_layout.groups.size() ||
        metadata.group_posting_size.size() != metadata.label_count ||
        metadata.label_size.size() != metadata.label_count ||
        metadata.label_offset.size() != metadata.label_count ||
        metadata.label_filter_bit.size() != metadata.original_label_count) {
        fail_graph(
            "coverage_cost requires dimensionally consistent grouped runtime metadata");
    }

    for (uint32_t group_id = 0; group_id < metadata.label_count; ++group_id) {
        const auto& labels = group_layout.groups[group_id];
        if (labels.size() > 64) {
            fail_graph(
                "coverage_cost group size exceeds original-label filter mask capacity");
        }
        for (size_t bit_idx = 0; bit_idx < labels.size(); ++bit_idx) {
            const uint32_t original_label = labels[bit_idx];
            if (original_label >= metadata.original_label_count ||
                group_layout.label_to_group[original_label] != group_id) {
                fail_graph(
                    "coverage_cost group layout contains an invalid original-label mapping");
            }
            const uint64_t expected_filter_bit = 1ULL << bit_idx;
            if (metadata.label_filter_bit[original_label] != expected_filter_bit) {
                fail_graph(
                    "coverage_cost original-label filter bit conflicts with group layout");
            }
        }
    }

    if (!metadata.original_label_cardinality.empty()) {
        if (metadata.original_label_cardinality.size() != metadata.original_label_count) {
            fail_graph(
                "coverage_cost original-label cardinality metadata has the wrong length");
        }
        for (uint32_t group_id = 0; group_id < metadata.label_count; ++group_id) {
            const auto& labels = group_layout.groups[group_id];
            uint64_t cardinality_sum = 0;
            for (uint32_t original_label : labels) {
                if (original_label >= metadata.original_label_cardinality.size() ||
                    group_layout.label_to_group[original_label] != group_id) {
                    fail_graph(
                        "coverage_cost original-label cardinality metadata conflicts with group layout");
                }
                const uint32_t cardinality =
                    metadata.original_label_cardinality[original_label];
                if (cardinality > metadata.group_posting_size[group_id]) {
                    fail_graph(
                        "coverage_cost original-label cardinality exceeds its group union size");
                }
                cardinality_sum = saturating_add_u64(cardinality_sum, cardinality);
            }
            if (labels.size() == 1 &&
                metadata.original_label_cardinality[labels.front()] !=
                    metadata.group_posting_size[group_id]) {
                fail_graph(
                    "coverage_cost singleton-group cardinality differs from its posting size");
            }
            if (!labels.empty() && metadata.group_posting_size[group_id] > 0 &&
                cardinality_sum < metadata.group_posting_size[group_id]) {
                fail_graph(
                    "coverage_cost original-label cardinalities cannot cover their group union");
            }
        }
        return;
    }

    std::vector<uint32_t> derived_cardinality(metadata.original_label_count, 0);
    uint64_t grouped_rows_scanned = 0;
    for (uint32_t group_id = 0; group_id < metadata.label_count; ++group_id) {
        const auto& labels = group_layout.groups[group_id];
        const uint32_t group_rows = metadata.group_posting_size[group_id];
        if (labels.empty()) {
            if (group_rows != 0) {
                fail_graph(
                    "coverage_cost cannot derive coverage for a nonempty posting with no group labels");
            }
            continue;
        }
        if (group_rows == 0) {
            continue;
        }

        const uint64_t* masks = nullptr;
        if (metadata.label_size[group_id] > 0) {
            if (metadata.label_size[group_id] != group_rows) {
                fail_graph(
                    "coverage_cost high-frequency group rows differ from the group union size");
            }
            const uint64_t begin = metadata.label_offset[group_id];
            const uint64_t end = begin + group_rows;
            if (end > metadata.row_filter_mask.size()) {
                fail_graph(
                    "coverage_cost high-frequency filter masks are missing or truncated");
            }
            masks = metadata.row_filter_mask.data() + begin;
        } else {
            if (group_id >= runtime.lowfreq_index.label_size.size() ||
                group_id >= runtime.lowfreq_index.label_offset.size() ||
                runtime.lowfreq_index.label_size[group_id] != group_rows) {
                fail_graph(
                    "coverage_cost low-frequency posting metadata is missing or inconsistent");
            }
            const uint64_t begin = runtime.lowfreq_index.label_offset[group_id];
            const uint64_t end = begin + group_rows;
            if (end > runtime.lowfreq_index.row_filter_mask.size()) {
                fail_graph(
                    "coverage_cost low-frequency filter masks are missing or truncated");
            }
            masks = runtime.lowfreq_index.row_filter_mask.data() + begin;
        }

        if (labels.size() == 1) {
            derived_cardinality[labels.front()] = group_rows;
            continue;
        }

        const uint64_t allowed_mask = labels.size() == 64
            ? std::numeric_limits<uint64_t>::max()
            : ((1ULL << labels.size()) - 1ULL);
        grouped_rows_scanned = saturating_add_u64(grouped_rows_scanned, group_rows);
        for (uint32_t row = 0; row < group_rows; ++row) {
            uint64_t mask = masks[row];
            if (mask == 0 || (mask & ~allowed_mask) != 0) {
                fail_graph(
                    "coverage_cost encountered an invalid grouped original-label filter mask");
            }
            while (mask != 0) {
                const uint32_t bit_idx = static_cast<uint32_t>(__builtin_ctzll(mask));
                uint32_t& cardinality = derived_cardinality[labels[bit_idx]];
                if (cardinality == std::numeric_limits<uint32_t>::max()) {
                    fail_graph("coverage_cost original-label cardinality overflow");
                }
                ++cardinality;
                mask &= mask - 1;
            }
        }
    }
    metadata.original_label_cardinality = std::move(derived_cardinality);
    std::fprintf(stdout,
                 "[runtime-or-route-planner] derived exact original-label cardinality "
                 "from runtime filter masks labels=%u grouped_rows_scanned=%llu\n",
                 metadata.original_label_count,
                 static_cast<unsigned long long>(grouped_rows_scanned));
}

bool resolve_route_label_candidate(const GraphRuntime& runtime,
                                   uint32_t query_idx,
                                   uint32_t original_label,
                                   RouteLabelCandidate& candidate)
{
    if (original_label >= runtime.group_layout.label_to_group.size()) {
        fail_graph("query " + std::to_string(query_idx) + " has label outside group mapping");
    }
    const uint32_t group = runtime.group_layout.label_to_group[original_label];
    if (group == UINT32_MAX || group >= runtime.metadata.label_count) {
        fail_graph("query " + std::to_string(query_idx) + " has unmapped runtime label");
    }
    if (original_label >= runtime.metadata.label_filter_bit.size()) {
        fail_graph("query " + std::to_string(query_idx) + " has label outside filter bit range");
    }
    const uint64_t filter_bit = runtime.metadata.label_filter_bit[original_label];
    if (filter_bit == 0) {
        fail_graph("query " + std::to_string(query_idx) + " has no valid filter bit");
    }

    const uint32_t posting_size = runtime.metadata.group_posting_size[group];
    const bool is_highfreq = runtime.metadata.label_size[group] > 0;
    const bool is_lowfreq = !is_highfreq &&
                            group < runtime.lowfreq_index.label_size.size() &&
                            runtime.lowfreq_index.label_size[group] > 0;
    if (posting_size == 0 && !is_lowfreq) {
        return false;
    }

    candidate.original_label = original_label;
    candidate.group = group;
    candidate.posting_size = posting_size;
    candidate.filter_bit = filter_bit;
    candidate.is_lowfreq = is_lowfreq;
    return true;
}

void attach_or_route_physical_cost(const GraphRuntime& runtime,
                                   uint32_t original_position,
                                   RouteLabelCandidate& candidate)
{
    candidate.original_position = original_position;
    if (candidate.is_lowfreq) {
        candidate.exact_lowfreq_rows = runtime.lowfreq_index.label_size[candidate.group];
        candidate.physical_cost = candidate.exact_lowfreq_rows;
        return;
    }

    candidate.highfreq_cost_proxy = saturating_multiply_u64(
        runtime.metadata.label_size[candidate.group], runtime.graph.cols);
    candidate.physical_cost = candidate.highfreq_cost_proxy;
}

void attach_or_route_relevance(const GraphRuntime& runtime,
                               uint32_t query_idx,
                               size_t label_idx,
                               RouteLabelCandidate& candidate)
{
    const SparseLabelMatrix& matrix = runtime.original_query_labels_matrix;
    if (matrix.source_tiers.empty() && matrix.relevance_ranks.empty()) {
        candidate.source_tier = QueryLabelSourceTier::Exact;
        candidate.relevance_rank = 0;
        return;
    }
    if (matrix.source_tiers.size() != matrix.indices.size() ||
        matrix.relevance_ranks.size() != matrix.indices.size() ||
        query_idx + 1 >= matrix.indptr.size()) {
        fail_graph("query label occurrence metadata is not CSR-aligned");
    }
    const uint64_t occurrence_idx = matrix.indptr[query_idx] + label_idx;
    if (occurrence_idx >= matrix.indptr[query_idx + 1] ||
        occurrence_idx >= matrix.indices.size()) {
        fail_graph("query label occurrence metadata index is outside its CSR row");
    }
    const uint8_t source_tier = matrix.source_tiers[occurrence_idx];
    if (source_tier > static_cast<uint8_t>(QueryLabelSourceTier::Fallback)) {
        fail_graph("query label occurrence metadata has an invalid source tier");
    }
    candidate.source_tier = static_cast<QueryLabelSourceTier>(source_tier);
    candidate.relevance_rank = matrix.relevance_ranks[occurrence_idx];
}

void attach_or_route_exact_label_cardinality(const GraphRuntime& runtime,
                                             RouteLabelCandidate& candidate)
{
    if (candidate.original_label >=
        runtime.metadata.original_label_cardinality.size()) {
        fail_graph(
            "coverage_cost route is missing exact original-label cardinality metadata");
    }
    candidate.exact_label_cardinality =
        runtime.metadata.original_label_cardinality[candidate.original_label];
    if (candidate.exact_label_cardinality == 0) {
        fail_graph(
            "coverage_cost routable label has zero exact original-label cardinality");
    }
}

bool or_route_physical_cost_less(const RouteLabelCandidate& left,
                                 const RouteLabelCandidate& right)
{
    if (left.physical_cost != right.physical_cost) {
        return left.physical_cost < right.physical_cost;
    }
    if (left.original_position != right.original_position) {
        return left.original_position < right.original_position;
    }
    return left.original_label < right.original_label;
}

bool or_route_coverage_retention_less(const RouteLabelCandidate& left,
                                      const RouteLabelCandidate& right)
{
    if (left.exact_label_cardinality != right.exact_label_cardinality) {
        return left.exact_label_cardinality > right.exact_label_cardinality;
    }
    return or_route_physical_cost_less(left, right);
}

bool or_route_relevance_coverage_retention_less(const RouteLabelCandidate& left,
                                                const RouteLabelCandidate& right)
{
    const bool left_fallback = left.source_tier == QueryLabelSourceTier::Fallback;
    const bool right_fallback = right.source_tier == QueryLabelSourceTier::Fallback;
    if (left_fallback != right_fallback) {
        return left_fallback < right_fallback;
    }
    if (left.relevance_rank != right.relevance_rank) {
        return left.relevance_rank < right.relevance_rank;
    }
    if (left.exact_label_cardinality != right.exact_label_cardinality) {
        return left.exact_label_cardinality > right.exact_label_cardinality;
    }
    return or_route_physical_cost_less(left, right);
}

void attach_or_route_predicted_latency(const OrRouteAccuracyCostModel& model,
                                       RouteLabelCandidate& candidate)
{
    double predicted = candidate.is_lowfreq
        ? model.lowfreq_intercept_us +
              model.lowfreq_per_row_us * static_cast<double>(candidate.exact_lowfreq_rows)
        : model.highfreq_intercept_us +
              model.highfreq_per_work_us * static_cast<double>(candidate.highfreq_cost_proxy);
    predicted = std::max(predicted, model.cost_floor_us);
    if (!std::isfinite(predicted) || predicted <= 0.0) {
        fail_graph("calibrated route latency prediction is non-finite or non-positive");
    }
    candidate.predicted_latency_us = predicted;
}

bool or_route_predicted_latency_less(const RouteLabelCandidate& left,
                                     const RouteLabelCandidate& right)
{
    if (left.predicted_latency_us != right.predicted_latency_us) {
        return left.predicted_latency_us < right.predicted_latency_us;
    }
    if (left.original_position != right.original_position) {
        return left.original_position < right.original_position;
    }
    return left.original_label < right.original_label;
}

void evaluate_marginal_accuracy_cost_candidate(
    const OrRouteAccuracyCostModel& model,
    const ActiveOrRouteSketches& sketches,
    const std::vector<uint64_t>& current_union,
    const std::vector<uint64_t>& candidate_sketch,
    RouteLabelCandidate& candidate,
    std::vector<uint64_t>& merged_union)
{
    merged_union = merge_or_route_bottom_k(current_union,
                                           candidate_sketch,
                                           static_cast<size_t>(model.sketch_k));
    const double current_estimate = estimate_or_route_kmv_cardinality(
        current_union, static_cast<size_t>(model.sketch_k), sketches.num_points());
    const double merged_estimate = estimate_or_route_kmv_cardinality(
        merged_union, static_cast<size_t>(model.sketch_k), sketches.num_points());
    double marginal_gain = merged_estimate - current_estimate;
    if (!std::isfinite(marginal_gain)) {
        fail_graph("KMV marginal coverage estimate is non-finite");
    }
    marginal_gain = std::max(0.0, marginal_gain);
    marginal_gain = std::min(marginal_gain,
                             static_cast<double>(candidate.exact_label_cardinality));
    candidate.estimated_marginal_coverage = marginal_gain;

    double prior = model.exact_prior;
    if (candidate.source_tier == QueryLabelSourceTier::Direct) {
        prior = model.direct_prior;
    } else if (candidate.source_tier == QueryLabelSourceTier::Fallback) {
        prior = model.fallback_prior;
        const uint32_t zero_based_rank = candidate.relevance_rank > 0
            ? candidate.relevance_rank - 1
            : 0;
        prior *= std::exp(-model.fallback_rank_decay * zero_based_rank);
    }
    const double base_coverage_fraction = std::clamp(
        static_cast<double>(candidate.exact_label_cardinality) /
            static_cast<double>(sketches.num_points()),
        0.0,
        1.0);
    const double marginal_coverage_fraction = std::clamp(
        marginal_gain / static_cast<double>(sketches.num_points()), 0.0, 1.0);
    candidate.quality = prior +
        model.base_coverage_weight *
            std::pow(base_coverage_fraction, model.marginal_coverage_exponent) +
        model.marginal_coverage_weight *
            std::pow(marginal_coverage_fraction, model.marginal_coverage_exponent);
    const double normalized_cost = std::max(candidate.predicted_latency_us,
                                            model.cost_floor_us) /
                                   model.latency_scale_us;
    candidate.utility = model.quality_weight * candidate.quality -
                        model.cost_weight * std::pow(normalized_cost, model.cost_exponent);
    if (!std::isfinite(candidate.quality) || !std::isfinite(candidate.utility)) {
        fail_graph("calibrated route quality/utility is non-finite");
    }
}

bool marginal_accuracy_cost_candidate_better(const RouteLabelCandidate& candidate,
                                             const RouteLabelCandidate& current_best)
{
    if (candidate.utility != current_best.utility) {
        return candidate.utility > current_best.utility;
    }
    if (candidate.quality != current_best.quality) {
        return candidate.quality > current_best.quality;
    }
    if (candidate.estimated_marginal_coverage !=
        current_best.estimated_marginal_coverage) {
        return candidate.estimated_marginal_coverage >
               current_best.estimated_marginal_coverage;
    }
    if (candidate.predicted_latency_us != current_best.predicted_latency_us) {
        return candidate.predicted_latency_us < current_best.predicted_latency_us;
    }
    if (candidate.relevance_rank != current_best.relevance_rank) {
        return candidate.relevance_rank < current_best.relevance_rank;
    }
    if (candidate.original_position != current_best.original_position) {
        return candidate.original_position < current_best.original_position;
    }
    return candidate.original_label < current_best.original_label;
}

std::vector<RouteLabelCandidate> retain_or_routes_by_marginal_accuracy_cost(
    std::vector<RouteLabelCandidate> candidates,
    ActiveOrRouteSketches& sketches,
    const OrRouteAccuracyCostModel& model,
    OrRoutePlannerProfile& profile,
    bool& query_was_capped)
{
    const bool has_any_cap = profile.total_cap > 0 || profile.highfreq_cap > 0;
    if (!has_any_cap) {
        std::sort(candidates.begin(), candidates.end(), or_route_predicted_latency_less);
        std::vector<uint64_t> union_sketch;
        for (RouteLabelCandidate& candidate : candidates) {
            const std::vector<uint64_t>& candidate_sketch = sketches.get(
                candidate.original_label, candidate.exact_label_cardinality);
            std::vector<uint64_t> merged;
            evaluate_marginal_accuracy_cost_candidate(
                model, sketches, union_sketch, candidate_sketch, candidate, merged);
            union_sketch = std::move(merged);
            profile.retained_estimated_marginal_gain_sum +=
                candidate.estimated_marginal_coverage;
            if (candidate.estimated_marginal_coverage == 0.0) {
                saturating_increment_u64(profile.retained_zero_marginal_routes);
            }
        }
        profile.estimated_union_coverage_sum += estimate_or_route_kmv_cardinality(
            union_sketch, static_cast<size_t>(model.sketch_k), sketches.num_points());
        return candidates;
    }

    std::vector<RouteLabelCandidate> retained;
    retained.reserve(candidates.size());
    std::vector<uint8_t> selected(candidates.size(), 0);
    std::vector<uint64_t> union_sketch;
    uint32_t retained_highfreq = 0;
    while (profile.total_cap == 0 || retained.size() < profile.total_cap) {
        bool guarded_candidate_exists = false;
        bool feasible_candidate_exists = false;
        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            if (selected[idx] ||
                (!candidates[idx].is_lowfreq && profile.highfreq_cap > 0 &&
                 retained_highfreq >= profile.highfreq_cap)) {
                continue;
            }
            feasible_candidate_exists = true;
            if (candidates[idx].source_tier != QueryLabelSourceTier::Fallback) {
                guarded_candidate_exists = true;
            }
        }
        if (!feasible_candidate_exists) {
            break;
        }

        size_t best_idx = candidates.size();
        std::vector<uint64_t> best_union;
        for (size_t idx = 0; idx < candidates.size(); ++idx) {
            RouteLabelCandidate& candidate = candidates[idx];
            if (selected[idx] ||
                (!candidate.is_lowfreq && profile.highfreq_cap > 0 &&
                 retained_highfreq >= profile.highfreq_cap) ||
                (guarded_candidate_exists &&
                 candidate.source_tier == QueryLabelSourceTier::Fallback)) {
                continue;
            }
            const std::vector<uint64_t>& candidate_sketch = sketches.get(
                candidate.original_label, candidate.exact_label_cardinality);
            std::vector<uint64_t> merged;
            evaluate_marginal_accuracy_cost_candidate(
                model, sketches, union_sketch, candidate_sketch, candidate, merged);
            if (best_idx == candidates.size() ||
                marginal_accuracy_cost_candidate_better(candidate, candidates[best_idx])) {
                best_idx = idx;
                best_union = std::move(merged);
            }
        }
        if (best_idx == candidates.size()) {
            break;
        }
        selected[best_idx] = 1;
        if (!candidates[best_idx].is_lowfreq) {
            ++retained_highfreq;
        }
        union_sketch = std::move(best_union);
        profile.retained_estimated_marginal_gain_sum +=
            candidates[best_idx].estimated_marginal_coverage;
        if (candidates[best_idx].estimated_marginal_coverage == 0.0) {
            saturating_increment_u64(profile.retained_zero_marginal_routes);
        }
        retained.push_back(candidates[best_idx]);
    }

    for (size_t idx = 0; idx < candidates.size(); ++idx) {
        if (selected[idx]) {
            continue;
        }
        query_was_capped = true;
        saturating_increment_u64(profile.capped_routes);
        if (profile.total_cap > 0 && retained.size() >= profile.total_cap) {
            saturating_increment_u64(profile.total_cap_capped_routes);
        } else if (!candidates[idx].is_lowfreq && profile.highfreq_cap > 0 &&
                   retained_highfreq >= profile.highfreq_cap) {
            saturating_increment_u64(profile.highfreq_cap_capped_routes);
        } else {
            saturating_increment_u64(profile.total_cap_capped_routes);
        }
    }
    profile.estimated_union_coverage_sum += estimate_or_route_kmv_cardinality(
        union_sketch, static_cast<size_t>(model.sketch_k), sketches.num_points());
    std::sort(retained.begin(), retained.end(), or_route_predicted_latency_less);
    return retained;
}

void add_or_route_cost_to_profile(OrRoutePlannerProfile& profile,
                                  const RouteLabelCandidate& candidate,
                                  bool emitted)
{
    if (candidate.is_lowfreq) {
        uint64_t& rows = emitted ? profile.emitted_exact_lowfreq_rows
                                 : profile.original_exact_lowfreq_rows;
        rows = saturating_add_u64(rows, candidate.exact_lowfreq_rows);
        return;
    }

    uint64_t& proxy = emitted ? profile.emitted_highfreq_cost_proxy
                              : profile.original_highfreq_cost_proxy;
    proxy = saturating_add_u64(proxy, candidate.highfreq_cost_proxy);
}

void add_or_route_cardinality_to_profile(OrRoutePlannerProfile& profile,
                                         const RouteLabelCandidate& candidate,
                                         bool emitted)
{
    uint64_t& sum = emitted ? profile.emitted_exact_label_cardinality_sum
                            : profile.original_exact_label_cardinality_sum;
    sum = saturating_add_u64(sum, candidate.exact_label_cardinality);
}

void add_or_route_relevance_to_profile(OrRoutePlannerProfile& profile,
                                       const RouteLabelCandidate& candidate,
                                       bool emitted)
{
    uint64_t* route_count = nullptr;
    switch (candidate.source_tier) {
    case QueryLabelSourceTier::Direct:
        route_count = emitted ? &profile.emitted_direct_source_routes
                              : &profile.original_direct_source_routes;
        break;
    case QueryLabelSourceTier::Fallback:
        route_count = emitted ? &profile.emitted_fallback_source_routes
                              : &profile.original_fallback_source_routes;
        break;
    case QueryLabelSourceTier::Exact:
    default:
        route_count = emitted ? &profile.emitted_exact_source_routes
                              : &profile.original_exact_source_routes;
        break;
    }
    saturating_increment_u64(*route_count);
    if (emitted) {
        profile.retained_relevance_rank_sum = saturating_add_u64(
            profile.retained_relevance_rank_sum, candidate.relevance_rank);
        profile.retained_relevance_rank_max = std::max(
            profile.retained_relevance_rank_max, candidate.relevance_rank);
    }
}

template <typename Comparator>
std::vector<RouteLabelCandidate> retain_or_routes_by_priority(
    std::vector<RouteLabelCandidate> candidates,
    OrRoutePlannerProfile& profile,
    bool& query_was_capped,
    uint32_t total_cap,
    Comparator comparator)
{
    // Route caps only consume the best prefix. Sorting the complete candidate
    // vector makes fuzzy expansion pay O(n log n) even when a query keeps
    // eight routes; partial_sort preserves the exact strict ordering while
    // reducing this to O(n log k). Keep the full-sort path when a separate
    // high-frequency cap can cause lower-ranked candidates to be retained.
    const bool can_partial_sort =
        total_cap > 0 &&
        profile.highfreq_cap == 0 &&
        total_cap < candidates.size();
    const size_t retained_limit = can_partial_sort
        ? static_cast<size_t>(total_cap)
        : candidates.size();
    if (can_partial_sort) {
        std::partial_sort(candidates.begin(),
                          candidates.begin() + retained_limit,
                          candidates.end(),
                          comparator);
    } else {
        std::sort(candidates.begin(), candidates.end(), comparator);
    }
    std::vector<RouteLabelCandidate> retained;
    retained.reserve(std::min(candidates.size(), retained_limit));
    uint64_t retained_highfreq = 0;
    for (size_t candidate_idx = 0; candidate_idx < candidates.size(); ++candidate_idx) {
        const RouteLabelCandidate& candidate = candidates[candidate_idx];
        if (total_cap > 0 && retained.size() >= total_cap) {
            saturating_increment_u64(profile.capped_routes);
            saturating_increment_u64(profile.total_cap_capped_routes);
            query_was_capped = true;
            continue;
        }
        if (!candidate.is_lowfreq && profile.highfreq_cap > 0 &&
            retained_highfreq >= profile.highfreq_cap) {
            saturating_increment_u64(profile.capped_routes);
            saturating_increment_u64(profile.highfreq_cap_capped_routes);
            query_was_capped = true;
            continue;
        }
        retained.push_back(candidate);
        if (!candidate.is_lowfreq) {
            ++retained_highfreq;
        }
        if (can_partial_sort && retained.size() == retained_limit) {
            query_was_capped = true;
            for (size_t remaining = candidate_idx + 1;
                 remaining < candidates.size();
                 ++remaining) {
                saturating_increment_u64(profile.capped_routes);
                saturating_increment_u64(profile.total_cap_capped_routes);
            }
            break;
        }
    }
    std::sort(retained.begin(), retained.end(), or_route_physical_cost_less);
    return retained;
}

std::vector<RouteLabelCandidate> retain_or_routes_by_exact_coverage(
    std::vector<RouteLabelCandidate> candidates,
    OrRoutePlannerProfile& profile,
    bool& query_was_capped,
    uint32_t total_cap)
{
    return retain_or_routes_by_priority(std::move(candidates),
                                        profile,
                                        query_was_capped,
                                        total_cap,
                                        or_route_coverage_retention_less);
}

std::vector<RouteLabelCandidate> retain_or_routes_by_relevance_coverage(
    std::vector<RouteLabelCandidate> candidates,
    OrRoutePlannerProfile& profile,
    bool& query_was_capped,
    uint32_t total_cap)
{
    return retain_or_routes_by_priority(std::move(candidates),
                                        profile,
                                        query_was_capped,
                                        total_cap,
                                        or_route_relevance_coverage_retention_less);
}

bool should_dual_route_and_labels(const GraphRuntime& runtime,
                                  uint32_t left_posting_size,
                                  uint32_t right_posting_size)
{
    if (left_posting_size == 0 || right_posting_size == 0) {
        return false;
    }
    const uint32_t smaller = std::min(left_posting_size, right_posting_size);
    const uint32_t larger = std::max(left_posting_size, right_posting_size);
    return static_cast<double>(larger) / static_cast<double>(smaller) <
           runtime.and_dual_route_posting_ratio_threshold;
}

uint32_t route_cap_from_env(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    const unsigned long parsed = std::strtoul(value, nullptr, 10);
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        return std::numeric_limits<uint32_t>::max();
    }
    return static_cast<uint32_t>(parsed);
}

bool env_flag_enabled(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return normalized != "0" && normalized != "false" && normalized != "off" && normalized != "no";
}

void coarsen_range_cover_bins(std::vector<uint8_t>& child_active,
                              std::vector<uint8_t>& parent_active,
                              const std::vector<uint32_t>& parent_labels)
{
    if (parent_active.empty() || parent_active.size() != parent_labels.size() ||
        child_active.size() % parent_active.size() != 0) {
        fail_graph("range hierarchy levels must have dense integer-multiple bin counts");
    }
    const size_t children_per_parent = child_active.size() / parent_active.size();
    if (children_per_parent == 0) {
        fail_graph("range hierarchy parent cannot have zero children");
    }
    for (size_t parent_bin = 0; parent_bin < parent_active.size(); ++parent_bin) {
        const size_t child_begin = parent_bin * children_per_parent;
        const size_t child_end = child_begin + children_per_parent;
        const bool all_children_active = std::all_of(
            child_active.begin() + child_begin,
            child_active.begin() + child_end,
            [](uint8_t active) { return active != 0; });
        if (!all_children_active || parent_labels[parent_bin] == UINT32_MAX) {
            continue;
        }
        std::fill(child_active.begin() + child_begin,
                  child_active.begin() + child_end,
                  0);
        parent_active[parent_bin] = 1;
    }
}

struct RangeCoverLayout {
    float domain_low = std::numeric_limits<float>::infinity();
    float domain_high = -std::numeric_limits<float>::infinity();
    uint32_t max_level = 0;
    std::vector<std::vector<uint32_t>> labels_by_level;
    std::vector<std::vector<float>> lowers_by_level, uppers_by_level;
};

RangeCoverLayout build_range_cover_layout(const GraphRuntime& runtime, uint32_t range_idx)
{
    RangeCoverLayout layout;
    auto& domain_low = layout.domain_low;
    auto& domain_high = layout.domain_high;
    for (uint32_t label = 0; label < runtime.label_intervals.has_interval.size(); ++label) {
        if (!runtime.label_intervals.has_interval[label] ||
            runtime.label_intervals.range_indices[label] != range_idx) {
            continue;
        }
        domain_low = std::min(domain_low, runtime.label_intervals.lower_bounds[label]);
        domain_high = std::max(domain_high, runtime.label_intervals.upper_bounds[label]);
    }
    if (!std::isfinite(domain_low) || !std::isfinite(domain_high) || domain_low >= domain_high) {
        fail_graph("pure-range query requires range_label_manifest_path with valid intervals");
    }

    auto& max_level = layout.max_level;
    bool have_any_level = false;
    for (uint32_t label = 0; label < runtime.label_intervals.has_interval.size(); ++label) {
        if (!range_route_label_usable(runtime, label, range_idx)) {
            continue;
        }
        const uint32_t level = runtime.label_intervals.level_indices[label];
        const uint32_t bin = runtime.label_intervals.bin_indices[label];
        if (level == UINT32_MAX || bin == UINT32_MAX) {
            continue;
        }
        max_level = std::max(max_level, level);
        have_any_level = true;
    }
    if (!have_any_level) {
        fail_graph("pure-range query requires usable range labels in manifest");
    }

    auto& labels_by_level = layout.labels_by_level;
    labels_by_level.resize(max_level + 1);
    auto& lowers_by_level = layout.lowers_by_level;
    lowers_by_level.resize(max_level + 1);
    auto& uppers_by_level = layout.uppers_by_level;
    uppers_by_level.resize(max_level + 1);
    for (uint32_t label = 0; label < runtime.label_intervals.has_interval.size(); ++label) {
        if (!range_route_label_usable(runtime, label, range_idx)) {
            continue;
        }
        const uint32_t level = runtime.label_intervals.level_indices[label];
        const uint32_t bin = runtime.label_intervals.bin_indices[label];
        if (level == UINT32_MAX || bin == UINT32_MAX) {
            continue;
        }
        if (labels_by_level[level].size() <= bin) {
            labels_by_level[level].resize(static_cast<size_t>(bin) + 1, UINT32_MAX);
            lowers_by_level[level].resize(static_cast<size_t>(bin) + 1, 0.0f);
            uppers_by_level[level].resize(static_cast<size_t>(bin) + 1, 0.0f);
        }
        labels_by_level[level][bin] = label;
        lowers_by_level[level][bin] = runtime.label_intervals.lower_bounds[label];
        uppers_by_level[level][bin] = runtime.label_intervals.upper_bounds[label];
    }

    return layout;
}

std::vector<uint32_t> plan_pure_range_cover_labels(const GraphRuntime& runtime,
                                                   uint32_t query_idx,
                                                   uint32_t range_idx,
                                                   const RangeCoverLayout* prepared = nullptr)
{
    std::vector<uint32_t> range_labels;
    if (query_idx >= runtime.original_query_ranges.rows) {
        fail_graph("pure-range planner query index out of range");
    }

    if (range_idx >= runtime.original_query_ranges.range_count) {
        fail_graph("pure-range planner range index out of range");
    }

    const size_t query_flat = runtime.original_query_ranges.flat_index(query_idx, range_idx);
    const uint8_t mode = runtime.original_query_ranges.modes[query_flat];
    if (!query_range_active(mode)) {
        return range_labels;
    }

    RangeCoverLayout owned;
    if (prepared == nullptr) owned = build_range_cover_layout(runtime, range_idx);
    const auto& layout = prepared ? *prepared : owned;
    const float domain_low = layout.domain_low, domain_high = layout.domain_high;
    const uint32_t max_level = layout.max_level;
    const auto& labels_by_level = layout.labels_by_level;
    const auto& lowers_by_level = layout.lowers_by_level;
    const auto& uppers_by_level = layout.uppers_by_level;
    float target_low = domain_low;
    float target_high = domain_high;
    if (mode == 1 || mode == 3) {
        target_low = std::max(target_low, runtime.original_query_ranges.lower_bounds[query_flat]);
    }
    if (mode == 2 || mode == 3) {
        target_high = std::min(target_high, runtime.original_query_ranges.upper_bounds[query_flat]);
    }
    if (!(target_low < target_high)) {
        return range_labels;
    }

    struct RangePartitionNode {
        uint32_t label = UINT32_MAX;
        uint32_t level = UINT32_MAX;
        uint32_t bin = UINT32_MAX;
        float lower = 0.0f;
        float upper = 0.0f;
    };

    const uint32_t finest_level = max_level;
    if (labels_by_level[finest_level].empty()) {
        fail_graph("pure-range query has no usable finest-level range labels");
    }

    std::vector<std::vector<uint8_t>> active_bins(max_level + 1);
    for (uint32_t level = 0; level <= max_level; ++level) {
        active_bins[level].assign(labels_by_level[level].size(), 0);
    }

    bool selected_any = false;
    for (uint32_t bin = 0; bin < labels_by_level[finest_level].size(); ++bin) {
        const uint32_t label = labels_by_level[finest_level][bin];
        if (label == UINT32_MAX) {
            continue;
        }
        const float lower = lowers_by_level[finest_level][bin];
        const float upper = uppers_by_level[finest_level][bin];
        if (!intervals_overlap(lower, upper, target_low, target_high)) {
            continue;
        }
        active_bins[finest_level][bin] = 1;
        selected_any = true;
    }
    if (!selected_any) {
        fail_graph("unable to construct pure-range cover for query " + std::to_string(query_idx));
    }

    const bool use_finest_cover = env_flag_enabled("GO_FANN_PURE_RANGE_FINEST_COVER");
    if (!use_finest_cover) {
        for (uint32_t level = finest_level; level > 0; --level) {
            if (active_bins[level].empty() || labels_by_level[level - 1].empty()) {
                continue;
            }
            coarsen_range_cover_bins(active_bins[level],
                                     active_bins[level - 1],
                                     labels_by_level[level - 1]);
        }
    }

    std::vector<RangePartitionNode> selected_nodes;
    for (uint32_t level = 0; level <= max_level; ++level) {
        for (uint32_t bin = 0; bin < active_bins[level].size(); ++bin) {
            if (!active_bins[level][bin]) {
                continue;
            }
            const uint32_t label = labels_by_level[level][bin];
            if (label == UINT32_MAX) {
                fail_graph("pure-range planner selected missing label for query " +
                             std::to_string(query_idx));
            }
            selected_nodes.push_back({label, level, bin, lowers_by_level[level][bin], uppers_by_level[level][bin]});
        }
    }

    if (selected_nodes.empty()) {
        fail_graph("pure-range planner selected no labels for query " + std::to_string(query_idx));
    }

    std::sort(selected_nodes.begin(), selected_nodes.end(),
              [](const RangePartitionNode& lhs, const RangePartitionNode& rhs) {
                  constexpr float kSortEps = 1e-6f;
                  if (std::fabs(lhs.lower - rhs.lower) > kSortEps) {
                      return lhs.lower < rhs.lower;
                  }
                  if (std::fabs(lhs.upper - rhs.upper) > kSortEps) {
                      return lhs.upper < rhs.upper;
                  }
                  if (lhs.level != rhs.level) {
                      return lhs.level < rhs.level;
                  }
                  return lhs.bin < rhs.bin;
              });

    constexpr float kCoverEps = 1e-6f;
    float covered_until = target_low;
    for (const auto& node : selected_nodes) {
        if (node.upper <= covered_until + kCoverEps) {
            continue;
        }
        if (node.lower > covered_until + kCoverEps) {
            fail_graph("canonical pure-range cover leaves a gap for query " + std::to_string(query_idx));
        }
        covered_until = std::max(covered_until, node.upper);
        range_labels.push_back(node.label);
    }
    if (covered_until < target_high - kCoverEps) {
        fail_graph("canonical pure-range cover did not reach target interval for query " +
                     std::to_string(query_idx));
    }

    return range_labels;
}

bool load_original_label_seed_sidecar(const std::string& path,
                                      const GraphMetadata& metadata,
                                      const GroupLayout& group_layout,
                                      uint64_t num_points,
                                      uint64_t graph_degree,
                                      int specificity_threshold,
                                      std::vector<uint32_t>& seed_by_original_label)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        return false;
    }

    GraphOriginalLabelSeedHeader header;
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        return false;
    }

    const bool compatible =
        header.version == 2 &&
        header.original_label_count == metadata.original_label_count &&
        header.group_count == metadata.label_count &&
        header.num_points == num_points &&
        header.graph_rows == metadata.total_rows &&
        header.graph_degree == graph_degree &&
        header.specificity_threshold == static_cast<uint64_t>(specificity_threshold);
    if (!compatible) {
        return false;
    }

    seed_by_original_label.resize(metadata.original_label_count);
    input.read(reinterpret_cast<char*>(seed_by_original_label.data()),
               seed_by_original_label.size() * sizeof(uint32_t));
    if (!input) {
        return false;
    }

    for (uint32_t original_label = 0; original_label < metadata.original_label_count; ++original_label) {
        const uint32_t seed = seed_by_original_label[original_label];
        if (!is_valid_original_label_seed(metadata, group_layout, original_label, seed)) {
            return false;
        }
    }
    return true;
}

void save_original_label_seed_sidecar(const std::string& path,
                                      const GraphMetadata& metadata,
                                      const GroupLayout& group_layout,
                                      uint64_t num_points,
                                      uint64_t graph_degree,
                                      int specificity_threshold,
                                      const std::vector<uint32_t>& seed_by_original_label)
{
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        fail_graph("unable to write original label seed sidecar " + path);
    }

    GraphOriginalLabelSeedHeader header;
    header.original_label_count = metadata.original_label_count;
    header.group_count = metadata.label_count;
    header.num_points = num_points;
    header.graph_rows = metadata.total_rows;
    header.graph_degree = graph_degree;
    header.specificity_threshold = static_cast<uint64_t>(specificity_threshold);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(seed_by_original_label.data()),
                 seed_by_original_label.size() * sizeof(uint32_t));
    if (!output) {
        fail_graph("failed to write original label seed sidecar " + path);
    }
}

struct HybridClauseView {
    uint32_t clause_id = UINT32_MAX;
    uint32_t top_branch_id = 0;
    HybridClauseLabels labels;
    bool uses_original_range = false;
};

HybridClauseView hybrid_clause_view(const HybridQueryClauseTable& table,
                                    uint64_t clause_idx, uint32_t query_idx = UINT32_MAX)
{
    if (clause_idx >= table.clause_count() || clause_idx > UINT32_MAX ||
        !table.descriptor_shape_valid() ||
        table.source_tiers.size() != table.labels.size() ||
        table.relevance_ranks.size() != table.labels.size())
        fail_graph("hybrid clause table is not size-aligned");
    HybridClauseView view;
    view.clause_id = static_cast<uint32_t>(clause_idx);
    view.top_branch_id = table.clause_branch(clause_idx, query_idx);
    view.uses_original_range = table.clause_uses_range(clause_idx, query_idx);
    view.labels = table.clause_labels(clause_idx, query_idx);
    if (!view.labels.valid) fail_graph("hybrid clause label references are invalid");
    return view;
}

// Each chunk owns only query outputs. Large graph/index/metadata inputs are
// referenced, never copied. The caller joins before publishing the final state.
struct RoutePreparationShared {
    GraphRuntime& source;
    std::once_flag cardinality_once;
    double cardinality_ms = 0.0;
    explicit RoutePreparationShared(GraphRuntime& runtime) : source(runtime) {}
};
struct RoutePreparationChunk : GraphQueryState {
    RoutePreparationShared& shared;
    const decltype(GraphRuntime::vector_scalar_type)& vector_scalar_type;
    const decltype(GraphRuntime::original_query_vectors)& original_query_vectors;
    const decltype(GraphRuntime::original_query_float_vectors)& original_query_float_vectors;
    const decltype(GraphRuntime::original_query_count)& original_query_count;
    const decltype(GraphRuntime::original_query_labels_matrix)& original_query_labels_matrix;
    const decltype(GraphRuntime::original_query_label_ops)& original_query_label_ops;
    const decltype(GraphRuntime::original_query_clauses)& original_query_clauses;
    const decltype(GraphRuntime::original_query_ranges)& original_query_ranges;
    const decltype(GraphRuntime::graph)& graph;
    const decltype(GraphRuntime::lowfreq_index)& lowfreq_index;
    const decltype(GraphRuntime::group_layout)& group_layout;
    const decltype(GraphRuntime::metadata)& metadata;
    const decltype(GraphRuntime::label_intervals)& label_intervals;
    const decltype(GraphRuntime::seed_local_id_by_label)& seed_local_id_by_label;
    const decltype(GraphRuntime::seed_by_original_label)& seed_by_original_label;
    const decltype(GraphRuntime::seed_selection_mode)& seed_selection_mode;
    const decltype(GraphRuntime::and_dual_route_posting_ratio_threshold)& and_dual_route_posting_ratio_threshold;
    const decltype(GraphRuntime::supported_query_route_providers)& supported_query_route_providers;
    explicit RoutePreparationChunk(RoutePreparationShared& value)
        : shared(value),
          vector_scalar_type(value.source.vector_scalar_type),
          original_query_vectors(value.source.original_query_vectors),
          original_query_float_vectors(value.source.original_query_float_vectors),
          original_query_count(value.source.original_query_count),
          original_query_labels_matrix(value.source.original_query_labels_matrix),
          original_query_label_ops(value.source.original_query_label_ops),
          original_query_clauses(value.source.original_query_clauses),
          original_query_ranges(value.source.original_query_ranges),
          graph(value.source.graph),
          lowfreq_index(value.source.lowfreq_index),
          group_layout(value.source.group_layout),
          metadata(value.source.metadata),
          label_intervals(value.source.label_intervals),
          seed_local_id_by_label(value.source.seed_local_id_by_label),
          seed_by_original_label(value.source.seed_by_original_label),
          seed_selection_mode(value.source.seed_selection_mode),
          and_dual_route_posting_ratio_threshold(value.source.and_dual_route_posting_ratio_threshold),
          supported_query_route_providers(value.source.supported_query_route_providers) {
        route_payload_preparation.deferred = value.source.route_payload_preparation.deferred;
        shared_route_query_vectors = value.source.shared_route_query_vectors;
    }
    operator const GraphRuntime&() const { return shared.source; }
};

void ensure_planner_cardinality(GraphRuntime& runtime) {
    const auto started = LoadClock::now();
    ensure_exact_original_label_cardinality(runtime);
    runtime.query_preparation_timing.static_label_cardinality_init_ms += ns_to_ms(elapsed_load_ns(started));
}
void ensure_planner_cardinality(RoutePreparationChunk& chunk) {
    std::call_once(chunk.shared.cardinality_once, [&] {
        const auto started = LoadClock::now();
        ensure_exact_original_label_cardinality(chunk.shared.source);
        chunk.shared.cardinality_ms = ns_to_ms(elapsed_load_ns(started));
    });
}

template <class Runtime>
void append_route_payload(Runtime& runtime, uint32_t original_query_idx,
                          uint32_t chosen_original_label, RouteResidualMode residual_mode,
                          const HybridClauseView* clause, bool suppress_range = false)
{
    for (uint32_t range_idx = 0; range_idx < runtime.original_query_ranges.range_count; ++range_idx) {
        const size_t src_flat = runtime.original_query_ranges.flat_index(original_query_idx, range_idx);
        const bool copy_range = clause == nullptr || clause->uses_original_range;
        runtime.query_ranges.modes.push_back(
            copy_range && !suppress_range ? runtime.original_query_ranges.modes[src_flat] : 0U);
        runtime.query_ranges.lower_bounds.push_back(
            copy_range ? runtime.original_query_ranges.lower_bounds[src_flat] : 0.0f);
        runtime.query_ranges.upper_bounds.push_back(
            copy_range ? runtime.original_query_ranges.upper_bounds[src_flat] : 0.0f);
    }
    runtime.query_ranges.rows += 1;

    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        const uint8_t* src = runtime.original_query_vectors.row_ptr(original_query_idx);
        if (!runtime.shared_route_query_vectors)
            runtime.query_vectors.data.insert(runtime.query_vectors.data.end(),
                                              src, src + runtime.original_query_vectors.cols);
    } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        const float* src = runtime.original_query_float_vectors.row_ptr(original_query_idx);
        if (!runtime.shared_route_query_vectors)
            runtime.query_float_vectors.data.insert(runtime.query_float_vectors.data.end(),
                                                    src, src + runtime.original_query_float_vectors.cols);
        runtime.query_float_vectors.rows += 1;
    } else {
        fail_graph("unsupported query vector scalar type while routing");
    }
    runtime.query_vectors.rows += 1;

    auto& routed_labels = runtime.query_labels_matrix.row_labels.emplace_back();
    if (residual_mode == RouteResidualMode::RemainingConjunction) {
        const uint32_t* labels_begin = nullptr;
        const uint32_t* labels_end = nullptr;
        uint64_t occurrence_begin = 0;
        const std::vector<uint8_t>* source_tiers = nullptr;
        const std::vector<uint32_t>* relevance_ranks = nullptr;
        if (clause != nullptr) {
            labels_begin = clause->labels.begin();
            labels_end = clause->labels.end();
            occurrence_begin = 0;
            source_tiers = &runtime.original_query_clauses.source_tiers;
            relevance_ranks = &runtime.original_query_clauses.relevance_ranks;
        } else {
            const SparseLabelRowSpan row = sparse_label_row_span(
                runtime.original_query_labels_matrix, original_query_idx);
            if (!row.valid) {
                fail_graph("original query label CSR is invalid while appending route");
            }
            labels_begin = row.begin;
            labels_end = row.end;
            occurrence_begin = runtime.original_query_labels_matrix.indptr[original_query_idx];
            source_tiers = &runtime.original_query_labels_matrix.source_tiers;
            relevance_ranks = &runtime.original_query_labels_matrix.relevance_ranks;
        }
        const size_t label_count = labels_begin == nullptr
            ? 0
            : static_cast<size_t>(labels_end - labels_begin);
        // A singleton selected label has no residual. Avoid allocating an
        // empty row for every emitted route; push_back handles other singletons.
        if (label_count > 1) {
            routed_labels.reserve(label_count);
        }
        const bool has_occurrence_metadata = clause != nullptr ? clause->labels.has_metadata :
            !source_tiers->empty() || !relevance_ranks->empty();
        for (size_t label_idx = 0; label_idx < label_count; ++label_idx) {
            const uint32_t label = labels_begin[label_idx];
            if (label == chosen_original_label) {
                continue;
            }
            routed_labels.push_back(label);
            if (has_occurrence_metadata && clause != nullptr) {
                runtime.query_labels_matrix.source_tiers.push_back(clause->labels.tier(label_idx));
                runtime.query_labels_matrix.relevance_ranks.push_back(clause->labels.rank(label_idx));
            } else if (has_occurrence_metadata) {
                const uint64_t occurrence_idx = occurrence_begin + label_idx;
                if (source_tiers->size() != relevance_ranks->size() ||
                    occurrence_idx >= source_tiers->size()) {
                    fail_graph("routed query occurrence metadata is not CSR-aligned");
                }
                runtime.query_labels_matrix.source_tiers.push_back(
                    (*source_tiers)[occurrence_idx]);
                runtime.query_labels_matrix.relevance_ranks.push_back(
                    (*relevance_ranks)[occurrence_idx]);
            }
        }
    }
    runtime.query_labels_matrix.indices.insert(runtime.query_labels_matrix.indices.end(),
                                               routed_labels.begin(),
                                               routed_labels.end());
    runtime.query_labels_matrix.indptr.push_back(runtime.query_labels_matrix.indices.size());
    runtime.query_labels_matrix.rows += 1;

}

template <class Runtime>
void append_route(Runtime& runtime,
                  uint32_t original_query_idx,
                  uint32_t chosen_original_label,
                  uint32_t chosen_group,
                  uint32_t chosen_group_size,
                  uint64_t chosen_filter_bit,
                  RouteResidualMode residual_mode = RouteResidualMode::RemainingConjunction,
                  const HybridClauseView* clause = nullptr,
                  QueryRouteProvider provider = QueryRouteProvider::Label)
{
    if ((runtime.supported_query_route_providers & query_route_provider_bit(provider)) == 0) {
        fail_graph("query " + std::to_string(original_query_idx) +
                     " requires an unsupported route provider " +
                     std::to_string(static_cast<uint8_t>(provider)));
    }
    runtime.route_to_original_query.push_back(original_query_idx);
    runtime.route_to_clause.push_back(clause == nullptr ? UINT32_MAX : clause->clause_id);
    if (runtime.route_clause_indptr.empty()) {
        runtime.route_clause_indptr.push_back(0);
    }
    if (clause != nullptr) {
        runtime.route_clause_ids.push_back(clause->clause_id);
    }
    runtime.route_clause_indptr.push_back(runtime.route_clause_ids.size());
    runtime.route_top_branch_ids.push_back(clause == nullptr ? 0U : clause->top_branch_id);
    runtime.route_enabled.push_back(1U);
    runtime.route_providers.push_back(provider);
    runtime.route_execution_modes.push_back(QueryRouteExecution::None);
    runtime.route_candidate_sources.push_back(QueryCandidateSource::None);
    runtime.query_group_ids.push_back(chosen_group);
    runtime.query_filter_labels.push_back(chosen_original_label);
    runtime.query_filter_bits.push_back(chosen_filter_bit);
    runtime.query_is_lowfreq.push_back(0);
    const bool has_residual_labels = residual_mode == RouteResidualMode::RemainingConjunction &&
        (clause != nullptr ? clause->labels.begin() != clause->labels.end() :
         (runtime.original_query_labels_matrix.indptr[original_query_idx] != runtime.original_query_labels_matrix.indptr[original_query_idx + 1]));
    if (runtime.route_payload_preparation.deferred) {
        runtime.route_payload_preparation.flags.push_back(static_cast<uint8_t>(residual_mode));
        runtime.route_payload_preparation.excluded_labels.push_back(chosen_original_label);
        // Rows describe planned slots; the payload arrays remain empty.
        ++runtime.query_vectors.rows;
        if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) ++runtime.query_float_vectors.rows;
        ++runtime.query_labels_matrix.rows;
        ++runtime.query_ranges.rows;
    } else {
        append_route_payload(runtime, original_query_idx, chosen_original_label, residual_mode, clause);
    }

    const uint32_t route_idx = static_cast<uint32_t>(runtime.route_to_original_query.size() - 1);
    if (provider == QueryRouteProvider::RangePosting) {
        runtime.route_execution_modes.back() = has_residual_labels
            ? QueryRouteExecution::HostCandidateStream
            : QueryRouteExecution::GpuResidentScan;
        runtime.route_candidate_sources.back() = QueryCandidateSource::RangePosting;
        runtime.start_local_ids.push_back(UINT32_MAX);
        return;
    }
    if (runtime.metadata.label_size[chosen_group] > 0) {
        uint32_t seed_local;
        if (runtime.seed_selection_mode == "label") {
            seed_local = runtime.seed_by_original_label[chosen_original_label];
        } else {
            seed_local = runtime.seed_local_id_by_label[chosen_group];
        }
        if (seed_local == UINT32_MAX || seed_local >= runtime.metadata.label_size[chosen_group]) {
            fail_graph("query " + std::to_string(original_query_idx) + " has no valid seed local row");
        }
        runtime.high_query_indices.push_back(route_idx);
        runtime.route_execution_modes.back() = QueryRouteExecution::GraphTraversal;
        runtime.start_local_ids.push_back(seed_local);
        return;
    }

    if (chosen_group < runtime.lowfreq_index.label_size.size() && runtime.lowfreq_index.label_size[chosen_group] > 0) {
        runtime.query_is_lowfreq.back() = 1;
        runtime.low_query_indices.push_back(route_idx);
        runtime.route_execution_modes.back() = QueryRouteExecution::GpuResidentScan;
        runtime.route_candidate_sources.back() = QueryCandidateSource::LabelPosting;
        runtime.start_local_ids.push_back(UINT32_MAX);
        return;
    }

    runtime.query_group_ids.back() = UINT32_MAX;
    runtime.query_filter_labels.back() = UINT32_MAX;
    runtime.query_filter_bits.back() = 0;
    runtime.start_local_ids.push_back(UINT32_MAX);
}

struct HybridClauseRoutePlan {
    HybridClauseView clause;
    std::vector<RouteLabelCandidate> label_routes;
    RouteLabelCandidate priority_candidate;
    bool has_priority_candidate = false;
    bool append_range_route = false;
    bool fuzzy_semantic_branch = false;
};

void attach_hybrid_clause_relevance(const GraphRuntime& runtime,
                                    const HybridClauseView& clause,
                                    size_t label_idx,
                                    RouteLabelCandidate& candidate)
{
    (void)runtime;
    if (!clause.labels.valid || label_idx >= clause.labels.count)
        fail_graph("hybrid clause occurrence metadata index is out of range");
    const uint8_t source_tier = clause.labels.tier(label_idx);
    if (source_tier > static_cast<uint8_t>(QueryLabelSourceTier::Fallback))
        fail_graph("hybrid clause occurrence metadata has an invalid source tier");
    candidate.source_tier = static_cast<QueryLabelSourceTier>(source_tier);
    candidate.relevance_rank = clause.labels.rank(label_idx);
}

RouteLabelCandidate singleton_clause_candidate(const GraphRuntime& runtime,
                                                uint32_t query_idx,
                                                const HybridClauseView& clause)
{
    RouteLabelCandidate metadata;
    attach_hybrid_clause_relevance(runtime, clause, 0, metadata);
    RouteLabelCandidate candidate;
    if (!resolve_route_label_candidate(runtime, query_idx, *clause.labels.begin(), candidate)) {
        fail_graph("hybrid query " + std::to_string(query_idx) +
                     " clause " + std::to_string(clause.clause_id) +
                     " produced no usable label route");
    }
    attach_or_route_physical_cost(runtime, 0, candidate);
    candidate.source_tier = metadata.source_tier;
    candidate.relevance_rank = metadata.relevance_rank;
    return candidate;
}

HybridClauseRoutePlan plan_hybrid_clause_routes(const GraphRuntime& runtime,
                                                uint32_t query_idx,
                                                const HybridClauseView& clause,
                                                const std::string& routing_mode,
                                                bool summary_only = false)
{
    HybridClauseRoutePlan plan;
    plan.clause = clause;
    const bool has_labels = clause.labels.begin() != clause.labels.end();
    if (clause.uses_original_range &&
        !query_has_any_active_range(runtime.original_query_ranges, query_idx)) {
        fail_graph("hybrid query " + std::to_string(query_idx) +
                     " has a range clause but no active runtime range");
    }
    if (!has_labels) {
        if (!clause.uses_original_range) {
            fail_graph("hybrid query " + std::to_string(query_idx) +
                         " produced an empty DNF clause");
        }
        plan.append_range_route = true;
        return plan;
    }

    plan.append_range_route =
        clause.uses_original_range &&
        (runtime.supported_query_route_providers & kQueryRouteProviderRangePosting) != 0;

    // Singleton DNF branches dominate fuzzy expansion. Keep their only route
    // inline in the priority candidate, including branches discarded by a cap.
    if (clause.labels.end() - clause.labels.begin() == 1) {
        plan.priority_candidate = singleton_clause_candidate(runtime, query_idx, clause);
        plan.has_priority_candidate = true;
        plan.fuzzy_semantic_branch = plan.priority_candidate.source_tier != QueryLabelSourceTier::Exact;
        return plan;
    }

    const size_t label_count = clause.labels.count;
    std::array<RouteLabelCandidate, 2> inline_candidates;
    std::vector<RouteLabelCandidate> overflow_candidates;
    if (label_count > inline_candidates.size()) overflow_candidates.resize(label_count);
    RouteLabelCandidate* routable_labels = label_count <= inline_candidates.size()
        ? inline_candidates.data() : overflow_candidates.data();
    size_t routable_count = 0;
    RouteLabelCandidate chosen_lowfreq_candidate;
    bool has_lowfreq_candidate = false;
    QueryLabelSourceTier branch_source_tier = QueryLabelSourceTier::Exact;
    uint32_t branch_relevance_rank = 0;
    for (const uint32_t* label = clause.labels.begin(); label != clause.labels.end(); ++label) {
        const size_t label_idx = static_cast<size_t>(label - clause.labels.begin());
        RouteLabelCandidate occurrence_metadata;
        attach_hybrid_clause_relevance(runtime, clause, label_idx, occurrence_metadata);
        plan.fuzzy_semantic_branch = plan.fuzzy_semantic_branch ||
            occurrence_metadata.source_tier != QueryLabelSourceTier::Exact;
        if (static_cast<uint8_t>(occurrence_metadata.source_tier) >
            static_cast<uint8_t>(branch_source_tier)) {
            branch_source_tier = occurrence_metadata.source_tier;
        }
        branch_relevance_rank = std::max(branch_relevance_rank,
                                         occurrence_metadata.relevance_rank);
        RouteLabelCandidate candidate;
        if (!resolve_route_label_candidate(runtime, query_idx, *label, candidate)) {
            continue;
        }
        attach_or_route_physical_cost(runtime, static_cast<uint32_t>(label_idx), candidate);
        candidate.source_tier = occurrence_metadata.source_tier;
        candidate.relevance_rank = occurrence_metadata.relevance_rank;
        if (candidate.is_lowfreq &&
            (!has_lowfreq_candidate ||
             candidate.posting_size < chosen_lowfreq_candidate.posting_size)) {
            chosen_lowfreq_candidate = candidate;
            has_lowfreq_candidate = true;
        }
        routable_labels[routable_count++] = candidate;
    }
    const RouteLabelCandidate* selected_begin = routable_labels;
    const RouteLabelCandidate* selected_end = routable_labels + routable_count;
    if (has_lowfreq_candidate) {
        selected_begin = &chosen_lowfreq_candidate;
        selected_end = selected_begin + 1;
    } else if (routing_mode == "per_label_multi_route" && routable_count != 0) {
        // Keep all entries in the final plan, but only its minimum for ranking.
    } else if (routable_count == 2 &&
               should_dual_route_and_labels(runtime,
                                            routable_labels[0].posting_size,
                                            routable_labels[1].posting_size)) {
        if (routable_labels[1].posting_size < routable_labels[0].posting_size)
            std::swap(routable_labels[0], routable_labels[1]);
    } else {
        const RouteLabelCandidate* chosen = nullptr;
        for (size_t i = 0; i < routable_count; ++i) {
            const auto& candidate = routable_labels[i];
            if (candidate.posting_size == 0) continue;
            if (chosen == nullptr || should_replace_route_choice(
                    routing_mode, candidate.posting_size, chosen->posting_size))
                chosen = &candidate;
        }
        selected_begin = chosen;
        selected_end = chosen ? chosen + 1 : nullptr;
    }
    if (selected_begin == selected_end) {
        fail_graph("hybrid query " + std::to_string(query_idx) +
                     " clause " + std::to_string(clause.clause_id) +
                     " produced no usable label route");
    }
    if (!summary_only) plan.label_routes.assign(selected_begin, selected_end);
    plan.priority_candidate = *std::min_element(selected_begin, selected_end, or_route_physical_cost_less);
    plan.priority_candidate.source_tier = branch_source_tier;
    plan.priority_candidate.relevance_rank = branch_relevance_rank;
    plan.has_priority_candidate = true;
    return plan;
}

template <class Runtime>
void append_hybrid_clause_route_plan(Runtime& runtime,
                                     uint32_t query_idx,
                                     const HybridClauseRoutePlan& plan)
{
    const auto append_label = [&](const RouteLabelCandidate& candidate) {
        append_route(runtime,
                     query_idx,
                     candidate.original_label,
                     candidate.group,
                     candidate.posting_size,
                     candidate.filter_bit,
                     RouteResidualMode::RemainingConjunction,
                     &plan.clause,
                     QueryRouteProvider::Label);
    };
    if (plan.label_routes.empty() && plan.has_priority_candidate) {
        append_label(plan.priority_candidate);
    } else {
        for (const RouteLabelCandidate& candidate : plan.label_routes) {
            append_label(candidate);
        }
    }
    if (plan.append_range_route) {
        append_route(runtime,
                     query_idx,
                     UINT32_MAX,
                     UINT32_MAX,
                     0,
                     0,
                     RouteResidualMode::RemainingConjunction,
                     &plan.clause,
                     QueryRouteProvider::RangePosting);
    }
}

// Only the ranking key survives for a retained branch. Full route plans
// are built after selection, so scratch scales with the cap, not expansion.
struct CompactClausePriority {
    uint64_t physical_cost;
    uint32_t position;
    uint32_t cardinality;
    uint32_t relevance;
    QueryLabelSourceTier tier;
};

// Optional: cap fuzzy atom alternatives before deriving any DNF
// clause view. Original expressions remain complete for exact predicates.
// Hybrid AND ranking uses the fuzzy label here, rather than the physical
// entry chosen after examining the conjunction, so recall must be evaluated.
template <class Runtime, class Better>
bool select_pre_dnf_fuzzy_positions(const Runtime& runtime, uint32_t query,
                                    uint32_t cap, Better better,
                                    std::vector<uint32_t>& positions)
{
    const char* enabled = std::getenv("GO_FANN_PRE_DNF_FUZZY_CAP");
    if (!enabled || std::strcmp(enabled, "1") != 0) return false;
    const auto& table = runtime.original_query_clauses;
    if (!table.virtual_descriptors || query >= table.expressions.size()) return false;
    const auto& expression = table.expressions[query];
    if (!expression.active) return false;
    size_t fuzzy_atom = 2;
    const size_t atom_count = expression.unary ? 1 : 2;
    for (size_t a = 0; a < atom_count; ++a) {
        if (!expression.atoms[a].alternatives) continue;
        if (fuzzy_atom != 2) return false; // Unsupported multiple fuzzy atoms.
        fuzzy_atom = a;
    }
    if (fuzzy_atom == 2) return false;
    const auto& atom = expression.atoms[fuzzy_atom];
    const auto& sibling = expression.atoms[1 - fuzzy_atom];
    if (!expression.unary && HybridQueryClauseTable::alternatives(sibling) != 1) return false;
    const uint32_t fuzzy_offset = !expression.unary && !expression.conjunction && fuzzy_atom == 1
        ? static_cast<uint32_t>(HybridQueryClauseTable::alternatives(expression.atoms[0])) : 0;
    std::vector<CompactClausePriority> retained;
    retained.reserve(std::min(cap, atom.count));
    for (uint32_t i = 0; i < atom.count; ++i) {
        const uint64_t slot = atom.begin + i;
        const uint32_t label = table.atom_labels.at(slot);
        const auto tier = static_cast<QueryLabelSourceTier>(table.atom_source_tiers.at(slot));
        const uint32_t position = fuzzy_offset + i;
        // Deduplicating identical conjunct labels takes the exact occurrence's
        // tier. Such a branch must stay outside the fuzzy budget as before.
        const bool duplicate_exact = !expression.unary && expression.conjunction && !sibling.range &&
            sibling.count == 1 && label == table.atom_labels.at(sibling.begin) &&
            table.atom_source_tiers.at(sibling.begin) == static_cast<uint8_t>(QueryLabelSourceTier::Exact);
        if (tier == QueryLabelSourceTier::Exact || duplicate_exact) {
            positions.push_back(position);
            continue;
        }
        RouteLabelCandidate candidate;
        if (!resolve_route_label_candidate(runtime, query, label, candidate))
            fail_graph("pre-DNF fuzzy cap encountered an unroutable label");
        attach_or_route_physical_cost(runtime, position, candidate);
        attach_or_route_exact_label_cardinality(runtime, candidate);
        const CompactClausePriority key{candidate.physical_cost, position,
            candidate.exact_label_cardinality, table.atom_relevance_ranks.at(slot), tier};
        if (retained.size() < cap) {
            retained.push_back(key);
            std::push_heap(retained.begin(), retained.end(), better);
        } else if (better(key, retained.front())) {
            std::pop_heap(retained.begin(), retained.end(), better);
            retained.back() = key;
            std::push_heap(retained.begin(), retained.end(), better);
        }
    }
    for (const auto& key : retained) positions.push_back(key.position);
    // OR's exact label or range sibling is mandatory and does not consume cap.
    if (!expression.unary && !expression.conjunction)
        positions.push_back(fuzzy_atom == 0 ? atom.count : 0);
    std::sort(positions.begin(), positions.end());
    return true;
}

template <class Runtime>
bool try_append_compact_clause_query(Runtime& runtime,
                                        uint32_t query_idx,
                                        const std::string& routing_mode,
                                        const OrRoutePlannerRuntimeSelection& selection,
                                        bool calibration_trace_enabled,
                                        bool& exact_cardinality_ready)
{
    const auto& table = runtime.original_query_clauses;
    const uint64_t begin = table.query_indptr[query_idx];
    const uint64_t end = table.query_indptr[static_cast<size_t>(query_idx) + 1];
    auto& profile = runtime.or_route_planner_profile;
    const uint32_t manifest_cap = query_idx < table.branch_caps.size() ? table.branch_caps[query_idx] : 0;
    const uint32_t cap = manifest_cap ? manifest_cap : profile.total_cap;
    const bool fuzzy_cap = manifest_cap && query_idx < table.branch_cap_is_fuzzy.size() &&
        table.branch_cap_is_fuzzy[query_idx];
    const auto policy = fuzzy_cap ? selection.fuzzy_cap_mode : selection.mode;
    if (begin > end || end > table.clause_count() ||
        !table.descriptor_shape_valid() ||
        end - begin <= 1 || cap == 0 || profile.highfreq_cap != 0 || calibration_trace_enabled ||
        policy == OrRouteOrderMode::MarginalAccuracyCost ||
        !table.descriptor_shape_valid()) return false;
    if (table.virtual_descriptors && table.expressions[query_idx].active) {
        const auto& expression = table.expressions[query_idx];
        if (!fuzzy_cap && (expression.atoms[0].range || expression.atoms[1].range)) return false;
        // A native expression has at most two literals per branch. The parent
        // validated occurrence arrays; scoring below validates each branch.
    } else for (uint64_t i = begin; i < end; ++i) {
        const auto labels = table.clause_labels(i, query_idx);
        if (!labels.valid || labels.count > 2 ||
            (table.clause_uses_range(i, query_idx) && !fuzzy_cap)) return false;
    }
    const bool relevance = (manifest_cap && !fuzzy_cap) ||
        policy == OrRouteOrderMode::RelevanceCoverageCost;
    const bool cardinality = relevance || policy == OrRouteOrderMode::CoverageCost;
    if (cardinality && !exact_cardinality_ready) {
        ensure_planner_cardinality(runtime);
        exact_cardinality_ready = true;
    }
    if (cardinality) profile.exact_cardinality_used = true;
    if (manifest_cap) {
        saturating_increment_u64(profile.manifest_cap_queries);
        profile.manifest_cap_max = std::max(profile.manifest_cap_max, manifest_cap);
    }
    const auto better = [&](const CompactClausePriority& a, const CompactClausePriority& b) {
        if (relevance) {
            const bool af = a.tier == QueryLabelSourceTier::Fallback;
            const bool bf = b.tier == QueryLabelSourceTier::Fallback;
            if (af != bf) return af < bf;
            if (a.relevance != b.relevance) return a.relevance < b.relevance;
        }
        if (cardinality && a.cardinality != b.cardinality) return a.cardinality > b.cardinality;
        if ((cardinality || policy == OrRouteOrderMode::CostAware) && a.physical_cost != b.physical_cost)
            return a.physical_cost < b.physical_cost;
        // Positions are unique within a query: this also resolves label ties.
        return a.position < b.position;
    };
    std::vector<CompactClausePriority> retained;
    retained.reserve(std::min<uint64_t>(cap, end - begin));
    std::vector<uint32_t> emitted_positions;
    uint64_t eligible = 0;
    std::vector<uint32_t> pre_dnf_positions;
    const bool early_cap = fuzzy_cap && cardinality &&
        select_pre_dnf_fuzzy_positions(runtime, query_idx, cap, better, pre_dnf_positions);
    const uint64_t considered = early_cap ? pre_dnf_positions.size() : end - begin;
    for (uint64_t ordinal = 0; ordinal < considered; ++ordinal) {
        const uint64_t i = begin + (early_cap ? pre_dnf_positions[ordinal] : ordinal);
        const auto clause = hybrid_clause_view(table, i, query_idx);
        const auto summary = plan_hybrid_clause_routes(runtime, query_idx, clause, routing_mode, true);
        auto candidate = summary.priority_candidate;
        if (summary.has_priority_candidate) {
            if (cardinality) attach_or_route_exact_label_cardinality(runtime, candidate);
            add_or_route_relevance_to_profile(profile, candidate, false);
            if (candidate.is_lowfreq) saturating_increment_u64(profile.original_lowfreq_routes);
            else saturating_increment_u64(profile.original_highfreq_routes);
            add_or_route_cost_to_profile(profile, candidate, false);
            if (cardinality) add_or_route_cardinality_to_profile(profile, candidate, false);
        } else {
            candidate.physical_cost = std::numeric_limits<uint64_t>::max();
            candidate.source_tier = QueryLabelSourceTier::Exact;
        }
        saturating_increment_u64(profile.original_routes);
        const uint32_t position = static_cast<uint32_t>(i - begin);
        if (fuzzy_cap && !summary.fuzzy_semantic_branch) {
            emitted_positions.push_back(position);
            continue;
        }
        ++eligible;
        const CompactClausePriority key{candidate.physical_cost, position,
            candidate.exact_label_cardinality, candidate.relevance_rank, candidate.source_tier};
        if (retained.size() < cap) {
            retained.push_back(key);
            std::push_heap(retained.begin(), retained.end(), better);
        } else if (better(key, retained.front())) {
            std::pop_heap(retained.begin(), retained.end(), better);
            retained.back() = key;
            std::push_heap(retained.begin(), retained.end(), better);
        }
    }
    if (fuzzy_cap && eligible == 0) fail_graph("fuzzy branch cap has no fuzzy semantic alternatives");
    const uint64_t early_discarded = early_cap ? end - begin - considered : 0;
    profile.original_routes = saturating_add_u64(profile.original_routes, early_discarded);
    const uint64_t discarded = eligible - retained.size() + early_discarded;
    // Original cost/relevance sums only cover inspected branches in early-cap
    // mode. Counts include omitted branches; no physical plan was built for them.
    profile.capped_routes = saturating_add_u64(profile.capped_routes, discarded);
    profile.total_cap_capped_routes = saturating_add_u64(profile.total_cap_capped_routes, discarded);
    if (discarded) saturating_increment_u64(profile.capped_queries);
    saturating_increment_u64(profile.queries);
    for (const auto& key : retained) emitted_positions.push_back(key.position);
    std::sort(emitted_positions.begin(), emitted_positions.end());
    for (uint32_t position : emitted_positions) {
        const auto clause = hybrid_clause_view(table, begin + position, query_idx);
        const auto plan = plan_hybrid_clause_routes(runtime, query_idx, clause, routing_mode);
        auto candidate = plan.priority_candidate;
        append_hybrid_clause_route_plan(runtime, query_idx, plan);
        saturating_increment_u64(profile.emitted_routes);
        if (plan.has_priority_candidate) {
            if (cardinality) attach_or_route_exact_label_cardinality(runtime, candidate);
            add_or_route_relevance_to_profile(profile, candidate, true);
            if (candidate.is_lowfreq) saturating_increment_u64(profile.emitted_lowfreq_routes);
            else saturating_increment_u64(profile.emitted_highfreq_routes);
            add_or_route_cost_to_profile(profile, candidate, true);
            if (cardinality) add_or_route_cardinality_to_profile(profile, candidate, true);
        }
    }
    return true;
}

template <class Runtime>
void plan_and_append_clause_query(Runtime& runtime,
                                  uint32_t query_idx,
                                  const std::string& routing_mode,
                                  const OrRoutePlannerRuntimeSelection& planner_selection,
                                  bool calibration_trace_enabled,
                                  bool& exact_cardinality_ready,
                                  bool compact_singletons)
{
    if (compact_singletons && try_append_compact_clause_query(runtime, query_idx,
            routing_mode, planner_selection, calibration_trace_enabled, exact_cardinality_ready)) return;
    const uint64_t clause_begin = runtime.original_query_clauses.query_indptr[query_idx];
    const uint64_t clause_end =
        runtime.original_query_clauses.query_indptr[static_cast<size_t>(query_idx) + 1];
    std::vector<HybridClauseRoutePlan> plans;
    plans.reserve(static_cast<size_t>(clause_end - clause_begin));
    bool has_range_clause = false;
    bool has_multi_route_clause = false;
    for (uint64_t clause_idx = clause_begin; clause_idx < clause_end; ++clause_idx) {
        const HybridClauseView clause =
            hybrid_clause_view(runtime.original_query_clauses, clause_idx, query_idx);
        HybridClauseRoutePlan plan =
            plan_hybrid_clause_routes(runtime, query_idx, clause, routing_mode);
        has_range_clause = has_range_clause || clause.uses_original_range;
        has_multi_route_clause = has_multi_route_clause || plan.label_routes.size() > 1;
        if (runtime.route_payload_preparation.deferred) {
            // Keep only the clause reference and ranking summary until the cap
            // is resolved. Physical alternatives are rebuilt for selected clauses.
            std::vector<RouteLabelCandidate>().swap(plan.label_routes);
        }
        plans.push_back(std::move(plan));
    }
    if (plans.empty()) {
        fail_graph("query " + std::to_string(query_idx) +
                     " has an empty DNF branch set");
    }
    if (plans.size() == 1) {
        const auto& plan = plans.front();
        if (runtime.route_payload_preparation.deferred)
            append_hybrid_clause_route_plan(runtime, query_idx,
                plan_hybrid_clause_routes(runtime, query_idx, plan.clause, routing_mode));
        else
            append_hybrid_clause_route_plan(runtime, query_idx, plan);
        return;
    }

    OrRoutePlannerProfile& profile = runtime.or_route_planner_profile;
    const uint32_t manifest_branch_cap =
        query_idx < runtime.original_query_clauses.branch_caps.size()
        ? runtime.original_query_clauses.branch_caps[query_idx]
        : 0;
    const bool manifest_cap_is_fuzzy =
        manifest_branch_cap > 0 &&
        query_idx < runtime.original_query_clauses.branch_cap_is_fuzzy.size() &&
        runtime.original_query_clauses.branch_cap_is_fuzzy[query_idx] != 0;
    const OrRouteOrderMode branch_policy_mode = manifest_cap_is_fuzzy
        ? planner_selection.fuzzy_cap_mode
        : planner_selection.mode;
    if (manifest_branch_cap > 0) {
        saturating_increment_u64(profile.manifest_cap_queries);
        profile.manifest_cap_max =
            std::max(profile.manifest_cap_max, manifest_branch_cap);
    }
    const uint32_t effective_total_cap =
        manifest_branch_cap > 0 ? manifest_branch_cap : profile.total_cap;
    const bool has_any_cap = effective_total_cap > 0 || profile.highfreq_cap > 0;
    if (has_range_clause && has_any_cap && !manifest_cap_is_fuzzy) {
        fail_graph(
            "range-bearing DNF branches cannot be capped by global/explicit OR caps; only the "
            "fuzzy semantic-term cap may select their fuzzy alternatives");
    }
    if (has_multi_route_clause && profile.highfreq_cap > 0) {
        fail_graph(
            "high-frequency caps are undefined for DNF branches with multiple physical routes");
    }
    if (branch_policy_mode == OrRouteOrderMode::MarginalAccuracyCost) {
        fail_graph(
            "marginal_accuracy_cost is unsupported for clause-based OR planning; use "
            "relevance_coverage_cost");
    }
    if (calibration_trace_enabled) {
        fail_graph("OR route calibration trace is unsupported for clause-based OR planning");
    }

    const bool policy_uses_cardinality =
        branch_policy_mode == OrRouteOrderMode::CoverageCost ||
        branch_policy_mode == OrRouteOrderMode::RelevanceCoverageCost;
    const bool cardinality_mode = has_any_cap &&
        (policy_uses_cardinality ||
         (manifest_branch_cap > 0 && !manifest_cap_is_fuzzy));
    if (cardinality_mode && !exact_cardinality_ready) {
        ensure_planner_cardinality(runtime);
        exact_cardinality_ready = true;
    }
    if (cardinality_mode) {
        profile.exact_cardinality_used = true;
    }

    std::vector<RouteLabelCandidate> branch_candidates;
    branch_candidates.reserve(plans.size());
    for (size_t plan_idx = 0; plan_idx < plans.size(); ++plan_idx) {
        RouteLabelCandidate candidate;
        if (plans[plan_idx].has_priority_candidate) {
            candidate = plans[plan_idx].priority_candidate;
            if (cardinality_mode) {
                attach_or_route_exact_label_cardinality(runtime, candidate);
            }
        } else {
            candidate.original_label = UINT32_MAX;
            candidate.group = UINT32_MAX;
            candidate.physical_cost = std::numeric_limits<uint64_t>::max();
            candidate.source_tier = QueryLabelSourceTier::Exact;
        }
        candidate.original_position = static_cast<uint32_t>(plan_idx);
        branch_candidates.push_back(candidate);

        saturating_increment_u64(profile.original_routes);
        if (plans[plan_idx].has_priority_candidate) {
            add_or_route_relevance_to_profile(profile, candidate, false);
            if (candidate.is_lowfreq) {
                saturating_increment_u64(profile.original_lowfreq_routes);
            } else {
                saturating_increment_u64(profile.original_highfreq_routes);
            }
            add_or_route_cost_to_profile(profile, candidate, false);
            if (cardinality_mode) {
                add_or_route_cardinality_to_profile(profile, candidate, false);
            }
        }
    }
    saturating_increment_u64(profile.queries);

    std::vector<RouteLabelCandidate> mandatory_candidates;
    if (manifest_cap_is_fuzzy) {
        std::vector<RouteLabelCandidate> fuzzy_candidates;
        fuzzy_candidates.reserve(branch_candidates.size());
        mandatory_candidates.reserve(branch_candidates.size());
        for (const RouteLabelCandidate& candidate : branch_candidates) {
            if (plans[candidate.original_position].fuzzy_semantic_branch) {
                fuzzy_candidates.push_back(candidate);
            } else {
                mandatory_candidates.push_back(candidate);
            }
        }
        if (fuzzy_candidates.empty()) {
            fail_graph("fuzzy branch cap has no fuzzy semantic alternatives");
        }
        branch_candidates = std::move(fuzzy_candidates);
    }

    bool query_was_capped = false;
    if (manifest_branch_cap > 0 && !manifest_cap_is_fuzzy) {
        branch_candidates = retain_or_routes_by_relevance_coverage(
            std::move(branch_candidates), profile, query_was_capped, effective_total_cap);
    } else if (has_any_cap && branch_policy_mode == OrRouteOrderMode::CoverageCost) {
        branch_candidates = retain_or_routes_by_exact_coverage(
            std::move(branch_candidates), profile, query_was_capped, effective_total_cap);
    } else if (has_any_cap &&
               branch_policy_mode == OrRouteOrderMode::RelevanceCoverageCost) {
        branch_candidates = retain_or_routes_by_relevance_coverage(
            std::move(branch_candidates), profile, query_was_capped, effective_total_cap);
    } else if (has_any_cap && branch_policy_mode == OrRouteOrderMode::CostAware) {
        std::sort(branch_candidates.begin(),
                  branch_candidates.end(),
                  or_route_physical_cost_less);
    }

    if (!cardinality_mode && has_any_cap) {
        std::vector<RouteLabelCandidate> retained;
        retained.reserve(branch_candidates.size());
        uint32_t retained_highfreq = 0;
        for (const RouteLabelCandidate& candidate : branch_candidates) {
            if (effective_total_cap > 0 && retained.size() >= effective_total_cap) {
                saturating_increment_u64(profile.capped_routes);
                saturating_increment_u64(profile.total_cap_capped_routes);
                query_was_capped = true;
                continue;
            }
            if (!candidate.is_lowfreq && candidate.original_label != UINT32_MAX &&
                profile.highfreq_cap > 0 && retained_highfreq >= profile.highfreq_cap) {
                saturating_increment_u64(profile.capped_routes);
                saturating_increment_u64(profile.highfreq_cap_capped_routes);
                query_was_capped = true;
                continue;
            }
            retained.push_back(candidate);
            if (!candidate.is_lowfreq && candidate.original_label != UINT32_MAX) {
                ++retained_highfreq;
            }
        }
        branch_candidates = std::move(retained);
    }
    branch_candidates.insert(branch_candidates.end(),
                             mandatory_candidates.begin(),
                             mandatory_candidates.end());
    if (branch_candidates.empty()) {
        fail_graph("clause-based OR query " + std::to_string(query_idx) +
                     " produced no usable branches");
    }
    std::sort(branch_candidates.begin(), branch_candidates.end(), [](const auto& left,
                                                                     const auto& right) {
        return left.original_position < right.original_position;
    });

    for (const RouteLabelCandidate& candidate : branch_candidates) {
        const HybridClauseRoutePlan& plan = plans[candidate.original_position];
        if (runtime.route_payload_preparation.deferred)
            append_hybrid_clause_route_plan(runtime, query_idx,
                plan_hybrid_clause_routes(runtime, query_idx, plan.clause, routing_mode));
        else
            append_hybrid_clause_route_plan(runtime, query_idx, plan);
        saturating_increment_u64(profile.emitted_routes);
        if (plan.has_priority_candidate) {
            add_or_route_relevance_to_profile(profile, candidate, true);
            if (candidate.is_lowfreq) {
                saturating_increment_u64(profile.emitted_lowfreq_routes);
            } else {
                saturating_increment_u64(profile.emitted_highfreq_routes);
            }
            add_or_route_cost_to_profile(profile, candidate, true);
            if (cardinality_mode) {
                add_or_route_cardinality_to_profile(profile, candidate, true);
            }
        }
    }
    if (query_was_capped) {
        saturating_increment_u64(profile.capped_queries);
    }
}

template <class Runtime>
void report_prepared_query_routes(const Runtime& runtime)
{
    if (runtime.original_query_clauses.clause_count() > 0) {
        uint64_t hybrid_query_count = 0;
        uint64_t label_route_count = 0;
        uint64_t range_posting_route_count = 0;
        uint64_t highfreq_route_count = 0;
        uint64_t lowfreq_route_count = 0;
        for (uint32_t query_idx = 0; query_idx < runtime.original_query_count; ++query_idx) {
            hybrid_query_count += runtime.original_query_clauses.has_clauses(query_idx) ? 1U : 0U;
        }
        for (size_t route_idx = 0; route_idx < runtime.route_providers.size(); ++route_idx) {
            if (runtime.route_providers[route_idx] == QueryRouteProvider::RangePosting) {
                ++range_posting_route_count;
                continue;
            }
            ++label_route_count;
            if (route_idx < runtime.query_is_lowfreq.size() &&
                runtime.query_is_lowfreq[route_idx] != 0) {
                ++lowfreq_route_count;
            } else if (route_idx < runtime.query_group_ids.size() &&
                       runtime.query_group_ids[route_idx] != UINT32_MAX) {
                ++highfreq_route_count;
            }
        }
        std::fprintf(stdout,
                     "[runtime-hybrid-routes] queries=%llu clauses=%llu routes=%llu "
                     "label_routes=%llu range_posting_routes=%llu highfreq_routes=%llu "
                     "lowfreq_routes=%llu missing_provider=0\n",
                     static_cast<unsigned long long>(hybrid_query_count),
                     static_cast<unsigned long long>(
                         runtime.original_query_clauses.clause_count()),
                     static_cast<unsigned long long>(runtime.route_providers.size()),
                     static_cast<unsigned long long>(label_route_count),
                     static_cast<unsigned long long>(range_posting_route_count),
                     static_cast<unsigned long long>(highfreq_route_count),
                     static_cast<unsigned long long>(lowfreq_route_count));
    }
    const OrRoutePlannerProfile& profile = runtime.or_route_planner_profile;
    if (profile.queries > 0) {
        const bool cardinality_mode = profile.exact_cardinality_used ||
                                      profile.mode == OrRouteOrderMode::CoverageCost ||
                                      profile.mode == OrRouteOrderMode::RelevanceCoverageCost ||
                                      profile.mode == OrRouteOrderMode::MarginalAccuracyCost;
        const char* cardinality_source = cardinality_mode
            ? "original_label_filter_masks_exact"
            : "disabled";
        std::fprintf(stdout,
                     "[runtime-or-route-planner] mode=%s queries=%llu total_cap=%u highfreq_cap=%u manifest_cap_queries=%llu manifest_cap_max=%u original_routes=%llu emitted_routes=%llu original_high=%llu emitted_high=%llu original_low=%llu emitted_low=%llu original_exact_source_routes=%llu emitted_exact_source_routes=%llu original_direct_source_routes=%llu emitted_direct_source_routes=%llu original_fallback_source_routes=%llu emitted_fallback_source_routes=%llu retained_relevance_rank_sum=%llu retained_relevance_rank_max=%u reordered_queries=%llu reordered_routes=%llu capped_queries=%llu capped_routes=%llu total_cap_capped=%llu highfreq_cap_capped=%llu original_lowfreq_rows=%llu emitted_lowfreq_rows=%llu original_highfreq_cost_proxy=%llu emitted_highfreq_cost_proxy=%llu original_exact_label_cardinality_sum=%llu emitted_exact_label_cardinality_sum=%llu sketch_labels=%llu sketch_hashes=%llu sketch_bytes=%llu sketch_sidecar_bytes_read=%llu sketch_build_rows=%llu estimated_union_coverage_sum=%.6f retained_estimated_marginal_gain_sum=%.6f retained_zero_marginal_routes=%llu original_predicted_latency_us_sum=%.6f emitted_predicted_latency_us_sum=%.6f emitted_quality_sum=%.6f emitted_utility_sum=%.6f calibration_id=%s highfreq_proxy=graph_rows_x_degree latency_unit=microseconds cardinality_source=%s cardinality_sum_overlap_counting=%u\n",
                     or_route_order_mode_name(profile.mode),
                     static_cast<unsigned long long>(profile.queries),
                     profile.total_cap,
                     profile.highfreq_cap,
                     static_cast<unsigned long long>(profile.manifest_cap_queries),
                     profile.manifest_cap_max,
                     static_cast<unsigned long long>(profile.original_routes),
                     static_cast<unsigned long long>(profile.emitted_routes),
                     static_cast<unsigned long long>(profile.original_highfreq_routes),
                     static_cast<unsigned long long>(profile.emitted_highfreq_routes),
                     static_cast<unsigned long long>(profile.original_lowfreq_routes),
                     static_cast<unsigned long long>(profile.emitted_lowfreq_routes),
                     static_cast<unsigned long long>(profile.original_exact_source_routes),
                     static_cast<unsigned long long>(profile.emitted_exact_source_routes),
                     static_cast<unsigned long long>(profile.original_direct_source_routes),
                     static_cast<unsigned long long>(profile.emitted_direct_source_routes),
                     static_cast<unsigned long long>(profile.original_fallback_source_routes),
                     static_cast<unsigned long long>(profile.emitted_fallback_source_routes),
                     static_cast<unsigned long long>(profile.retained_relevance_rank_sum),
                     profile.retained_relevance_rank_max,
                     static_cast<unsigned long long>(profile.reordered_queries),
                     static_cast<unsigned long long>(profile.reordered_routes),
                     static_cast<unsigned long long>(profile.capped_queries),
                     static_cast<unsigned long long>(profile.capped_routes),
                     static_cast<unsigned long long>(profile.total_cap_capped_routes),
                     static_cast<unsigned long long>(profile.highfreq_cap_capped_routes),
                     static_cast<unsigned long long>(profile.original_exact_lowfreq_rows),
                     static_cast<unsigned long long>(profile.emitted_exact_lowfreq_rows),
                     static_cast<unsigned long long>(profile.original_highfreq_cost_proxy),
                     static_cast<unsigned long long>(profile.emitted_highfreq_cost_proxy),
                     static_cast<unsigned long long>(profile.original_exact_label_cardinality_sum),
                     static_cast<unsigned long long>(profile.emitted_exact_label_cardinality_sum),
                     static_cast<unsigned long long>(profile.sketch_labels),
                     static_cast<unsigned long long>(profile.sketch_hashes),
                     static_cast<unsigned long long>(profile.sketch_bytes),
                     static_cast<unsigned long long>(profile.sketch_sidecar_bytes_read),
                     static_cast<unsigned long long>(profile.sketch_build_rows),
                     profile.estimated_union_coverage_sum,
                     profile.retained_estimated_marginal_gain_sum,
                     static_cast<unsigned long long>(profile.retained_zero_marginal_routes),
                     profile.original_predicted_latency_us_sum,
                     profile.emitted_predicted_latency_us_sum,
                     profile.emitted_quality_sum,
                     profile.emitted_utility_sum,
                     profile.calibration_id.empty() ? "none" : profile.calibration_id.c_str(),
                     cardinality_source,
                     cardinality_mode ? 1U : 0U);
    }
}

template <class Runtime>
void prepare_query_route_chunk(Runtime& runtime,
                                  int dim,
                                  const std::string& routing_mode,
                                  const std::vector<uint8_t>& original_query_route_mask,
                                  uint32_t query_begin, uint32_t query_end,
                                  const std::vector<std::optional<RangeCoverLayout>>& cover_layouts,
                                  bool report)
{
    // Shared input is immutable while workers run. The parent validates once.
    if constexpr (std::is_same_v<Runtime, GraphRuntime>) {
        if (runtime.original_query_vectors.cols != static_cast<uint32_t>(dim)) {
            fail_graph("query vector dimension mismatch: file dim=" +
                         std::to_string(runtime.original_query_vectors.cols) +
                         ", configured dim=" + std::to_string(dim));
        }
        if (runtime.original_query_vectors.rows != runtime.original_query_labels_matrix.rows) {
            fail_graph("query vector count does not match query label rows");
        }
        validate_query_occurrence_metadata(runtime.original_query_labels_matrix,
                                           "original query label occurrence metadata");
        if (runtime.original_query_clauses.rows != 0 &&
            (runtime.original_query_clauses.rows != runtime.original_query_count ||
             runtime.original_query_clauses.query_indptr.size() !=
                 static_cast<size_t>(runtime.original_query_count) + 1 ||
             (!runtime.original_query_clauses.branch_caps.empty() &&
              runtime.original_query_clauses.branch_caps.size() !=
                  runtime.original_query_count) ||
             (!runtime.original_query_clauses.branch_cap_is_fuzzy.empty() &&
              runtime.original_query_clauses.branch_cap_is_fuzzy.size() !=
                  runtime.original_query_count) ||
             !runtime.original_query_clauses.descriptor_shape_valid())) {
            fail_graph("original hybrid query clause table is not query-aligned");
        }
        if (!original_query_route_mask.empty() &&
            original_query_route_mask.size() != runtime.original_query_count) {
            fail_graph("original query route mask is not query-aligned");
        }
    }

    runtime.route_payload_preparation.flags.clear();
    runtime.route_payload_preparation.excluded_labels.clear();
    runtime.query_vectors.rows = 0;
    runtime.query_vectors.cols = runtime.original_query_vectors.cols;
    runtime.query_vectors.data.clear();
    if (!runtime.route_payload_preparation.deferred && !runtime.shared_route_query_vectors) runtime.query_vectors.data.reserve(static_cast<size_t>(query_end - query_begin) * runtime.original_query_vectors.cols * (routing_mode == "per_label_multi_route" ? 2 : 1));
    runtime.query_float_vectors.rows = 0;
    runtime.query_float_vectors.cols = runtime.original_query_float_vectors.cols;
    runtime.query_float_vectors.data.clear();
    if (!runtime.route_payload_preparation.deferred && !runtime.shared_route_query_vectors) runtime.query_float_vectors.data.reserve(static_cast<size_t>(query_end - query_begin) * runtime.original_query_float_vectors.cols * (routing_mode == "per_label_multi_route" ? 2 : 1));
    runtime.query_labels_matrix.rows = 0;
    runtime.query_labels_matrix.cols = runtime.original_query_labels_matrix.cols;
    runtime.query_labels_matrix.indptr.clear();
    runtime.query_labels_matrix.indices.clear();
    runtime.query_labels_matrix.row_labels.clear();
    runtime.query_labels_matrix.source_tiers.clear();
    runtime.query_labels_matrix.relevance_ranks.clear();
    runtime.query_labels_matrix.indptr.push_back(0);
    runtime.query_group_ids.clear();
    runtime.query_filter_labels.clear();
    runtime.query_filter_bits.clear();
    runtime.query_ranges.rows = 0;
    runtime.query_ranges.range_count = runtime.original_query_ranges.range_count;
    runtime.query_ranges.modes.clear();
    runtime.query_ranges.lower_bounds.clear();
    runtime.query_ranges.upper_bounds.clear();
    runtime.start_local_ids.clear();
    runtime.query_is_lowfreq.clear();
    runtime.route_to_original_query.clear();
    runtime.route_to_clause.clear();
    runtime.route_clause_indptr.assign(1, 0);
    runtime.route_clause_ids.clear();
    runtime.route_top_branch_ids.clear();
    runtime.route_enabled.clear();
    runtime.route_providers.clear();
    runtime.route_execution_modes.clear();
    runtime.route_candidate_sources.clear();
    runtime.route_range_posting_begins.clear();
    runtime.route_range_posting_sizes.clear();
    runtime.route_range_posting_indices.clear();
    runtime.range_graph_and_original_queries.clear();
    runtime.range_graph_or_original_queries.clear();
    runtime.range_graph_clause_indptr.clear();
    runtime.range_graph_clause_ids.clear();
    runtime.range_access_planner_profile = RangeAccessPlannerProfile{};
    runtime.high_query_indices.clear();
    runtime.low_query_indices.clear();
    const OrRoutePlannerRuntimeSelection planner_selection =
        snapshot_or_route_planner_selection();
    runtime.or_route_planner_profile = OrRoutePlannerProfile{};
    const bool has_fuzzy_branch_cap = std::find(
        runtime.original_query_clauses.branch_cap_is_fuzzy.begin(),
        runtime.original_query_clauses.branch_cap_is_fuzzy.end(),
        static_cast<uint8_t>(1)) !=
        runtime.original_query_clauses.branch_cap_is_fuzzy.end();
    runtime.or_route_planner_profile.mode = has_fuzzy_branch_cap
        ? planner_selection.fuzzy_cap_mode
        : planner_selection.mode;
    runtime.or_route_planner_profile.total_cap =
        route_cap_from_env("GO_FANN_OR_ROUTE_TOTAL_CAP");
    runtime.or_route_planner_profile.highfreq_cap =
        route_cap_from_env("GO_FANN_OR_HIGHFREQ_CAP");
    const bool calibration_trace_enabled =
        !planner_selection.config.calibration_trace_path.empty();
    if (calibration_trace_enabled &&
        (runtime.or_route_planner_profile.total_cap > 0 ||
         runtime.or_route_planner_profile.highfreq_cap > 0)) {
        fail_graph(
            "OR route calibration trace requires GO_FANN_OR_ROUTE_TOTAL_CAP and "
            "GO_FANN_OR_HIGHFREQ_CAP to both be 0");
    }
    std::vector<OrRouteCandidateTraceRow> calibration_trace_rows;
    if (calibration_trace_enabled) {
        calibration_trace_rows.reserve(runtime.original_query_labels_matrix.indices.size());
    }
    bool exact_cardinality_ready = false;
    std::unique_ptr<ActiveOrRouteSketches> active_or_route_sketches;
    const char* compact_env = std::getenv("GO_FANN_COMPACT_CLAUSE_PLANNER");
    if (compact_env && std::strcmp(compact_env, "0") != 0 && std::strcmp(compact_env, "1") != 0)
        fail_graph("GO_FANN_COMPACT_CLAUSE_PLANNER must be 0 or 1");
    const bool compact_singletons = !compact_env || std::strcmp(compact_env, "0") != 0;
    for (uint32_t query_idx = query_begin; query_idx < query_end; ++query_idx) {
        if (!original_query_route_mask.empty() &&
            original_query_route_mask[query_idx] == 0) {
            continue;
        }
        if (runtime.original_query_clauses.has_clauses(query_idx)) {
            plan_and_append_clause_query(runtime,
                                         query_idx,
                                         routing_mode,
                                         planner_selection,
                                         calibration_trace_enabled,
                                         exact_cardinality_ready,
                                         compact_singletons);
            continue;
        }
        const auto span = sparse_label_row_span(runtime.original_query_labels_matrix, query_idx);
        if (!span.valid) fail_graph("original query has invalid label CSR");
        struct LabelsView {
            const uint32_t* first;
            const uint32_t* last;
            bool empty() const { return first == last; }
            size_t size() const { return empty() ? 0 : static_cast<size_t>(last - first); }
            const uint32_t* begin() const { return first; }
            const uint32_t* end() const { return last; }
            uint32_t operator[](size_t i) const { return first[i]; }
        } labels{span.begin, span.end};
        const bool has_active_range = query_has_any_active_range(runtime.original_query_ranges, query_idx);
        if (labels.empty()) {
            if (!has_active_range) {
                fail_graph("query " + std::to_string(query_idx) +
                             " has empty label row and no active range; pure-empty queries are unsupported");
            }
            if (has_active_range) {
                const bool range_posting_supported =
                    (runtime.supported_query_route_providers &
                     kQueryRouteProviderRangePosting) != 0;
                if (range_posting_supported) {
                    append_route(runtime,
                                 query_idx,
                                 UINT32_MAX,
                                 UINT32_MAX,
                                 0,
                                 0,
                                 RouteResidualMode::None,
                                 nullptr,
                                 QueryRouteProvider::RangePosting);
                }
                if ((runtime.supported_query_route_providers &
                     kQueryRouteProviderLabel) == 0) {
                    continue;
                }
                for (uint32_t range_idx = 0; range_idx < runtime.original_query_ranges.range_count; ++range_idx) {
                    const size_t query_flat = runtime.original_query_ranges.flat_index(query_idx, range_idx);
                    if (!query_range_active(runtime.original_query_ranges.modes[query_flat])) {
                        continue;
                    }
                    bool appended_route_for_dim = false;
                    const std::vector<uint32_t> range_route_labels =
                        plan_pure_range_cover_labels(runtime, query_idx, range_idx,
                            range_idx < cover_layouts.size() && cover_layouts[range_idx]
                                ? &*cover_layouts[range_idx] : nullptr);
                    for (uint32_t range_label : range_route_labels) {
                        if (range_label >= runtime.group_layout.label_to_group.size()) {
                            fail_graph("pure-range query " + std::to_string(query_idx) +
                                         " produced label outside group mapping");
                        }
                        const uint32_t group = runtime.group_layout.label_to_group[range_label];
                        if (group == UINT32_MAX || group >= runtime.metadata.label_count) {
                            fail_graph("pure-range query " + std::to_string(query_idx) +
                                         " produced unmapped runtime label");
                        }
                        const uint64_t filter_bit = runtime.metadata.label_filter_bit[range_label];
                        if (filter_bit == 0) {
                            fail_graph("pure-range query " + std::to_string(query_idx) +
                                         " produced label without valid filter bit");
                        }
                        const uint32_t posting_size = runtime.metadata.group_posting_size[group];
                        const bool has_lowfreq = group < runtime.lowfreq_index.label_size.size() &&
                                                 runtime.lowfreq_index.label_size[group] > 0;
                        if (posting_size == 0 && !has_lowfreq) {
                            continue;
                        }
                        append_route(runtime,
                                     query_idx,
                                     range_label,
                                     group,
                                     posting_size,
                                     filter_bit);
                        appended_route_for_dim = true;
                    }
                    if (!appended_route_for_dim) {
                        fail_graph("pure-range query " + std::to_string(query_idx) +
                                     " produced no usable routes for range dimension " +
                                     std::to_string(range_idx));
                    }
                }
            }
            continue;
        }

        const QueryLabelOp label_op = query_idx < runtime.original_query_label_ops.size()
            ? runtime.original_query_label_ops[query_idx]
            : QueryLabelOp::And;
        if (label_op == QueryLabelOp::Or) {
            OrRoutePlannerProfile& profile = runtime.or_route_planner_profile;
            const uint32_t manifest_branch_cap =
                query_idx < runtime.original_query_clauses.branch_caps.size()
                ? runtime.original_query_clauses.branch_caps[query_idx]
                : 0;
            if (manifest_branch_cap > 0) {
                saturating_increment_u64(profile.manifest_cap_queries);
                profile.manifest_cap_max =
                    std::max(profile.manifest_cap_max, manifest_branch_cap);
            }
            uint32_t effective_total_cap = profile.total_cap;
            if (manifest_branch_cap > 0 &&
                (effective_total_cap == 0 || manifest_branch_cap < effective_total_cap)) {
                effective_total_cap = manifest_branch_cap;
            }
            if (planner_selection.mode == OrRouteOrderMode::MarginalAccuracyCost &&
                (profile.total_cap != planner_selection.model.selection_route_cap ||
                 profile.highfreq_cap != planner_selection.model.selection_highfreq_cap)) {
                fail_graph(
                    "marginal_accuracy_cost runtime caps do not match the calibrated model: "
                    "GO_FANN_OR_ROUTE_TOTAL_CAP=" + std::to_string(profile.total_cap) +
                    " model=" + std::to_string(planner_selection.model.selection_route_cap) +
                    ", GO_FANN_OR_HIGHFREQ_CAP=" + std::to_string(profile.highfreq_cap) +
                    " model=" + std::to_string(planner_selection.model.selection_highfreq_cap));
            }
            const bool cardinality_mode =
                manifest_branch_cap > 0 ||
                planner_selection.mode == OrRouteOrderMode::CoverageCost ||
                planner_selection.mode == OrRouteOrderMode::RelevanceCoverageCost ||
                planner_selection.mode == OrRouteOrderMode::MarginalAccuracyCost;
            const bool exact_cardinality_required =
                cardinality_mode || calibration_trace_enabled;
            if (cardinality_mode) {
                profile.exact_cardinality_used = true;
            }
            if (exact_cardinality_required &&
                !exact_cardinality_ready) {
                ensure_planner_cardinality(runtime);
                exact_cardinality_ready = true;
            }
            if (planner_selection.mode == OrRouteOrderMode::MarginalAccuracyCost &&
                !active_or_route_sketches) {
                active_or_route_sketches = std::make_unique<ActiveOrRouteSketches>(
                    runtime, planner_selection, profile);
            }
            saturating_increment_u64(profile.queries);
            std::vector<RouteLabelCandidate> planned_routes;
            planned_routes.reserve(labels.size());
            std::vector<uint32_t> original_positions;
            original_positions.reserve(labels.size());
            for (size_t label_idx = 0; label_idx < labels.size(); ++label_idx) {
                RouteLabelCandidate candidate;
                if (!resolve_route_label_candidate(runtime, query_idx, labels[label_idx], candidate)) {
                    continue;
                }
                attach_or_route_physical_cost(
                    runtime, static_cast<uint32_t>(label_idx), candidate);
                attach_or_route_relevance(runtime, query_idx, label_idx, candidate);
                if (exact_cardinality_required) {
                    attach_or_route_exact_label_cardinality(runtime, candidate);
                }
                if (planner_selection.mode == OrRouteOrderMode::MarginalAccuracyCost) {
                    attach_or_route_predicted_latency(planner_selection.model, candidate);
                    profile.original_predicted_latency_us_sum +=
                        candidate.predicted_latency_us;
                }
                planned_routes.push_back(candidate);
                original_positions.push_back(candidate.original_position);
                saturating_increment_u64(profile.original_routes);
                if (candidate.is_lowfreq) {
                    saturating_increment_u64(profile.original_lowfreq_routes);
                } else {
                    saturating_increment_u64(profile.original_highfreq_routes);
                }
                add_or_route_cost_to_profile(profile, candidate, false);
                add_or_route_relevance_to_profile(profile, candidate, false);
                if (cardinality_mode) {
                    add_or_route_cardinality_to_profile(profile, candidate, false);
                }
            }
            if (planned_routes.empty()) {
                fail_graph("OR query " + std::to_string(query_idx) +
                             " produced no usable routes");
            }
            bool query_was_capped = false;
            if (manifest_branch_cap > 0 &&
                planner_selection.mode != OrRouteOrderMode::MarginalAccuracyCost) {
                planned_routes = retain_or_routes_by_relevance_coverage(
                    std::move(planned_routes),
                    profile,
                    query_was_capped,
                    effective_total_cap);
            } else if (planner_selection.mode == OrRouteOrderMode::CoverageCost) {
                planned_routes = retain_or_routes_by_exact_coverage(
                    std::move(planned_routes),
                    profile,
                    query_was_capped,
                    effective_total_cap);
            } else if (planner_selection.mode == OrRouteOrderMode::RelevanceCoverageCost) {
                planned_routes = retain_or_routes_by_relevance_coverage(
                    std::move(planned_routes),
                    profile,
                    query_was_capped,
                    effective_total_cap);
            } else if (planner_selection.mode == OrRouteOrderMode::MarginalAccuracyCost) {
                planned_routes = retain_or_routes_by_marginal_accuracy_cost(
                    std::move(planned_routes),
                    *active_or_route_sketches,
                    planner_selection.model,
                    profile,
                    query_was_capped);
            } else if (planner_selection.mode == OrRouteOrderMode::CostAware) {
                std::sort(planned_routes.begin(),
                          planned_routes.end(),
                          or_route_physical_cost_less);
            }
            std::vector<uint32_t> comparison_positions = original_positions;
            if (cardinality_mode) {
                comparison_positions.clear();
                comparison_positions.reserve(planned_routes.size());
                for (const RouteLabelCandidate& candidate : planned_routes) {
                    comparison_positions.push_back(candidate.original_position);
                }
                std::sort(comparison_positions.begin(), comparison_positions.end());
            }
            uint64_t reordered_for_query = 0;
            for (size_t route_idx = 0; route_idx < planned_routes.size(); ++route_idx) {
                if (planned_routes[route_idx].original_position != comparison_positions[route_idx]) {
                    ++reordered_for_query;
                }
            }
            if (reordered_for_query > 0) {
                saturating_increment_u64(profile.reordered_queries);
                profile.reordered_routes =
                    saturating_add_u64(profile.reordered_routes, reordered_for_query);
            }

            uint64_t appended_total_count = 0;
            uint64_t appended_highfreq_count = 0;
            bool appended_route = false;
            for (const RouteLabelCandidate& candidate : planned_routes) {
                if (!cardinality_mode &&
                    profile.total_cap > 0 && appended_total_count >= profile.total_cap) {
                    saturating_increment_u64(profile.capped_routes);
                    saturating_increment_u64(profile.total_cap_capped_routes);
                    query_was_capped = true;
                    continue;
                }
                if (!cardinality_mode &&
                    !candidate.is_lowfreq &&
                    profile.highfreq_cap > 0 &&
                    appended_highfreq_count >= profile.highfreq_cap) {
                    saturating_increment_u64(profile.capped_routes);
                    saturating_increment_u64(profile.highfreq_cap_capped_routes);
                    query_was_capped = true;
                    continue;
                }
                if (calibration_trace_enabled &&
                    runtime.route_to_original_query.size() >=
                        std::numeric_limits<uint32_t>::max()) {
                    fail_graph("OR route calibration trace route id exceeds uint32 range");
                }
                const uint32_t routed_route_id =
                    static_cast<uint32_t>(runtime.route_to_original_query.size());
                append_route(runtime,
                             query_idx,
                             candidate.original_label,
                             candidate.group,
                             candidate.posting_size,
                             candidate.filter_bit,
                             RouteResidualMode::None);
                if (calibration_trace_enabled) {
                    OrRouteCandidateTraceRow trace_row;
                    trace_row.original_query_id = query_idx;
                    trace_row.routed_route_id = routed_route_id;
                    trace_row.label = candidate.original_label;
                    trace_row.tier = candidate.source_tier;
                    trace_row.rank = candidate.relevance_rank;
                    trace_row.exact_label_cardinality = candidate.exact_label_cardinality;
                    trace_row.is_lowfreq = candidate.is_lowfreq;
                    trace_row.lowfreq_rows = candidate.exact_lowfreq_rows;
                    trace_row.highfreq_graph_rows = candidate.is_lowfreq
                        ? 0
                        : runtime.metadata.label_size[candidate.group];
                    trace_row.highfreq_degree = candidate.is_lowfreq
                        ? 0
                        : runtime.graph.cols;
                    trace_row.highfreq_work = candidate.is_lowfreq
                        ? 0
                        : candidate.highfreq_cost_proxy;
                    calibration_trace_rows.push_back(trace_row);
                }
                ++appended_total_count;
                saturating_increment_u64(profile.emitted_routes);
                if (candidate.is_lowfreq) {
                    saturating_increment_u64(profile.emitted_lowfreq_routes);
                } else {
                    ++appended_highfreq_count;
                    saturating_increment_u64(profile.emitted_highfreq_routes);
                }
                add_or_route_cost_to_profile(profile, candidate, true);
                add_or_route_relevance_to_profile(profile, candidate, true);
                if (cardinality_mode) {
                    add_or_route_cardinality_to_profile(profile, candidate, true);
                }
                if (planner_selection.mode == OrRouteOrderMode::MarginalAccuracyCost) {
                    profile.emitted_predicted_latency_us_sum += candidate.predicted_latency_us;
                    profile.emitted_quality_sum += candidate.quality;
                    profile.emitted_utility_sum += candidate.utility;
                }
                appended_route = true;
            }
            if (query_was_capped) {
                saturating_increment_u64(profile.capped_queries);
            }
            if (!appended_route) {
                fail_graph("OR query " + std::to_string(query_idx) +
                             " produced no usable routes");
            }
            continue;
        }

        std::vector<RouteLabelCandidate> routable_labels;
        routable_labels.reserve(labels.size());
        RouteLabelCandidate chosen_lowfreq_candidate;
        bool has_lowfreq_candidate = false;

        for (uint32_t original_label : labels) {
            RouteLabelCandidate candidate;
            if (!resolve_route_label_candidate(runtime, query_idx, original_label, candidate)) {
                continue;
            }
            if (candidate.is_lowfreq &&
                (!has_lowfreq_candidate || candidate.posting_size < chosen_lowfreq_candidate.posting_size)) {
                chosen_lowfreq_candidate = candidate;
                has_lowfreq_candidate = true;
            }
            routable_labels.push_back(candidate);
        }

        if (has_lowfreq_candidate) {
            append_route(runtime,
                         query_idx,
                         chosen_lowfreq_candidate.original_label,
                         chosen_lowfreq_candidate.group,
                         chosen_lowfreq_candidate.posting_size,
                         chosen_lowfreq_candidate.filter_bit);
            continue;
        }

        if (routing_mode == "per_label_multi_route") {
            for (const RouteLabelCandidate& candidate : routable_labels) {
                append_route(runtime,
                             query_idx,
                             candidate.original_label,
                             candidate.group,
                             candidate.posting_size,
                             candidate.filter_bit);
            }
            continue;
        }

        if (routable_labels.size() == 2 &&
            should_dual_route_and_labels(runtime,
                                         routable_labels[0].posting_size,
                                         routable_labels[1].posting_size)) {
            const RouteLabelCandidate* first = &routable_labels[0];
            const RouteLabelCandidate* second = &routable_labels[1];
            if (second->posting_size < first->posting_size) {
                std::swap(first, second);
            }
            append_route(runtime,
                         query_idx,
                         first->original_label,
                         first->group,
                         first->posting_size,
                         first->filter_bit);
            append_route(runtime,
                         query_idx,
                         second->original_label,
                         second->group,
                         second->posting_size,
                         second->filter_bit);
            continue;
        }

        uint32_t chosen_original_label = UINT32_MAX;
        uint32_t chosen_group = UINT32_MAX;
        uint32_t chosen_group_size = UINT32_MAX;
        uint64_t chosen_filter_bit = 0;

        for (const RouteLabelCandidate& candidate : routable_labels) {
            if (candidate.posting_size == 0) {
                continue;
            }
            if (chosen_original_label == UINT32_MAX ||
                should_replace_route_choice(routing_mode, candidate.posting_size, chosen_group_size)) {
                chosen_original_label = candidate.original_label;
                chosen_group = candidate.group;
                chosen_group_size = candidate.posting_size;
                chosen_filter_bit = candidate.filter_bit;
            }
        }

        if (chosen_original_label != UINT32_MAX) {
            append_route(runtime,
                         query_idx,
                         chosen_original_label,
                         chosen_group,
                         chosen_group_size,
                         chosen_filter_bit);
        }
    }
    if (calibration_trace_enabled) {
        write_or_route_calibration_trace(planner_selection.config.calibration_trace_path,
                                         planner_selection.config.calibration_run_id,
                                         calibration_trace_rows);
    }
    if (report) report_prepared_query_routes(runtime);
}

void validate_and_prepare_queries(GraphRuntime& runtime, int dim,
                                  const std::string& routing_mode,
                                  const std::vector<uint8_t>& original_query_route_mask, bool defer_payloads)
{
    runtime.route_payload_preparation.deferred = defer_payloads;
    const auto config = snapshot_query_preparation_config();
    const auto selection = snapshot_or_route_planner_selection();
    const uint32_t rows = runtime.original_query_vectors.rows;
    if (!config.optimized || selection.mode == OrRouteOrderMode::MarginalAccuracyCost ||
        !selection.config.calibration_trace_path.empty()) {
        prepare_query_route_chunk(runtime, dim, routing_mode, original_query_route_mask, 0, rows, {}, true);
        return;
    }
    // Construct range lookup tables online, after receiving the batch. Their
    // cost stays inside route_generation_ms and there is no on-disk query plan.
    std::vector<std::optional<RangeCoverLayout>> layouts(runtime.original_query_ranges.range_count);
    if ((runtime.supported_query_route_providers & kQueryRouteProviderLabel) != 0) {
        for (uint32_t query = 0; query < rows; ++query) {
            if (!original_query_route_mask.empty() && query < original_query_route_mask.size() && !original_query_route_mask[query]) continue;
            if (runtime.original_query_clauses.has_clauses(query) ||
                query >= runtime.original_query_labels_matrix.rows ||
                runtime.original_query_labels_matrix.indptr[query] != runtime.original_query_labels_matrix.indptr[query + 1]) continue;
            for (uint32_t range = 0; range < layouts.size(); ++range) {
                if (!layouts[range] && query < runtime.original_query_ranges.rows &&
                    query_range_active(runtime.original_query_ranges.modes[runtime.original_query_ranges.flat_index(query, range)]))
                    layouts[range] = build_range_cover_layout(runtime, range);
            }
        }
    }
    const size_t workers = go_fann::query_prepare::worker_count(config.threads, rows);
    if (workers <= 1 || rows < 128) {
        prepare_query_route_chunk(runtime, dim, routing_mode, original_query_route_mask, 0, rows, layouts, true);
        return;
    }
    // Validate/reset once before launching, including original mask/CSR shape.
    prepare_query_route_chunk(runtime, dim, routing_mode, original_query_route_mask, 0, 0, layouts, false);
    RoutePreparationShared shared{runtime};
    const size_t chunk_count = std::min<size_t>(rows, workers * 4);
    std::vector<std::unique_ptr<RoutePreparationChunk>> chunks(chunk_count);
    go_fann::query_prepare::parallel_tasks(chunk_count, static_cast<uint32_t>(workers), [&](size_t i) {
        auto chunk = std::make_unique<RoutePreparationChunk>(shared);
        const uint32_t begin = static_cast<uint64_t>(rows) * i / chunk_count;
        const uint32_t end = static_cast<uint64_t>(rows) * (i + 1) / chunk_count;
        prepare_query_route_chunk(*chunk, dim, routing_mode, original_query_route_mask, begin, end, layouts, false);
        chunks[i] = std::move(chunk);
    });
    runtime.query_preparation_timing.static_label_cardinality_init_ms += shared.cardinality_ms;
    const auto concatenate = [&](auto& destination, auto member) {
        size_t count = 0;
        for (const auto& chunk : chunks) count += (chunk.get()->*member).size();
        destination.reserve(count);
        for (auto& chunk : chunks) {
            auto& source = chunk.get()->*member;
            destination.insert(destination.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
        }
    };
    concatenate(runtime.query_group_ids, &GraphQueryState::query_group_ids);
    concatenate(runtime.query_filter_labels, &GraphQueryState::query_filter_labels);
    concatenate(runtime.query_filter_bits, &GraphQueryState::query_filter_bits);
    concatenate(runtime.start_local_ids, &GraphQueryState::start_local_ids);
    concatenate(runtime.query_is_lowfreq, &GraphQueryState::query_is_lowfreq);
    concatenate(runtime.route_to_original_query, &GraphQueryState::route_to_original_query);
    concatenate(runtime.route_to_clause, &GraphQueryState::route_to_clause);
    concatenate(runtime.route_clause_ids, &GraphQueryState::route_clause_ids);
    concatenate(runtime.route_top_branch_ids, &GraphQueryState::route_top_branch_ids);
    concatenate(runtime.route_enabled, &GraphQueryState::route_enabled);
    concatenate(runtime.route_providers, &GraphQueryState::route_providers);
    concatenate(runtime.route_execution_modes, &GraphQueryState::route_execution_modes);
    concatenate(runtime.route_candidate_sources, &GraphQueryState::route_candidate_sources);
    const auto concatenate_nested = [&](auto& destination, auto getter) {
        size_t count = 0;
        for (const auto& chunk : chunks) count += getter(*chunk).size();
        destination.reserve(count);
        for (auto& chunk : chunks) {
            auto& source = getter(*chunk);
            destination.insert(destination.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
        }
    };
    concatenate_nested(runtime.route_payload_preparation.excluded_labels, [](auto& chunk) -> auto& { return chunk.route_payload_preparation.excluded_labels; });
    concatenate_nested(runtime.route_payload_preparation.flags, [](auto& chunk) -> auto& { return chunk.route_payload_preparation.flags; });
    concatenate_nested(runtime.query_vectors.data, [](auto& chunk) -> auto& { return chunk.query_vectors.data; });
    concatenate_nested(runtime.query_float_vectors.data, [](auto& chunk) -> auto& { return chunk.query_float_vectors.data; });
    concatenate_nested(runtime.query_labels_matrix.indices, [](auto& chunk) -> auto& { return chunk.query_labels_matrix.indices; });
    concatenate_nested(runtime.query_labels_matrix.row_labels, [](auto& chunk) -> auto& { return chunk.query_labels_matrix.row_labels; });
    concatenate_nested(runtime.query_labels_matrix.source_tiers, [](auto& chunk) -> auto& { return chunk.query_labels_matrix.source_tiers; });
    concatenate_nested(runtime.query_labels_matrix.relevance_ranks, [](auto& chunk) -> auto& { return chunk.query_labels_matrix.relevance_ranks; });
    concatenate_nested(runtime.query_ranges.modes, [](auto& chunk) -> auto& { return chunk.query_ranges.modes; });
    concatenate_nested(runtime.query_ranges.lower_bounds, [](auto& chunk) -> auto& { return chunk.query_ranges.lower_bounds; });
    concatenate_nested(runtime.query_ranges.upper_bounds, [](auto& chunk) -> auto& { return chunk.query_ranges.upper_bounds; });
    uint64_t route_offset = 0, label_offset = 0, clause_offset = 0;
    for (const auto& chunk : chunks) {
        for (size_t i = 1; i < chunk->query_labels_matrix.indptr.size(); ++i)
            runtime.query_labels_matrix.indptr.push_back(label_offset + chunk->query_labels_matrix.indptr[i]);
        for (size_t i = 1; i < chunk->route_clause_indptr.size(); ++i)
            runtime.route_clause_indptr.push_back(clause_offset + chunk->route_clause_indptr[i]);
        for (uint32_t id : chunk->high_query_indices) runtime.high_query_indices.push_back(static_cast<uint32_t>(route_offset + id));
        for (uint32_t id : chunk->low_query_indices) runtime.low_query_indices.push_back(static_cast<uint32_t>(route_offset + id));
        route_offset += chunk->query_vectors.rows;
        label_offset += chunk->query_labels_matrix.indices.size();
        clause_offset += chunk->route_clause_ids.size();
        auto& dst = runtime.or_route_planner_profile;
        const auto& src = chunk->or_route_planner_profile;
        dst.queries = saturating_add_u64(dst.queries, src.queries);
        dst.original_routes = saturating_add_u64(dst.original_routes, src.original_routes);
        dst.emitted_routes = saturating_add_u64(dst.emitted_routes, src.emitted_routes);
        dst.original_highfreq_routes = saturating_add_u64(dst.original_highfreq_routes, src.original_highfreq_routes);
        dst.original_lowfreq_routes = saturating_add_u64(dst.original_lowfreq_routes, src.original_lowfreq_routes);
        dst.emitted_highfreq_routes = saturating_add_u64(dst.emitted_highfreq_routes, src.emitted_highfreq_routes);
        dst.emitted_lowfreq_routes = saturating_add_u64(dst.emitted_lowfreq_routes, src.emitted_lowfreq_routes);
        dst.original_exact_source_routes = saturating_add_u64(dst.original_exact_source_routes, src.original_exact_source_routes);
        dst.emitted_exact_source_routes = saturating_add_u64(dst.emitted_exact_source_routes, src.emitted_exact_source_routes);
        dst.original_direct_source_routes = saturating_add_u64(dst.original_direct_source_routes, src.original_direct_source_routes);
        dst.emitted_direct_source_routes = saturating_add_u64(dst.emitted_direct_source_routes, src.emitted_direct_source_routes);
        dst.original_fallback_source_routes = saturating_add_u64(dst.original_fallback_source_routes, src.original_fallback_source_routes);
        dst.emitted_fallback_source_routes = saturating_add_u64(dst.emitted_fallback_source_routes, src.emitted_fallback_source_routes);
        dst.retained_relevance_rank_sum = saturating_add_u64(dst.retained_relevance_rank_sum, src.retained_relevance_rank_sum);
        dst.retained_relevance_rank_max = std::max(dst.retained_relevance_rank_max, src.retained_relevance_rank_max);
        dst.reordered_queries = saturating_add_u64(dst.reordered_queries, src.reordered_queries);
        dst.reordered_routes = saturating_add_u64(dst.reordered_routes, src.reordered_routes);
        dst.capped_queries = saturating_add_u64(dst.capped_queries, src.capped_queries);
        dst.capped_routes = saturating_add_u64(dst.capped_routes, src.capped_routes);
        dst.total_cap_capped_routes = saturating_add_u64(dst.total_cap_capped_routes, src.total_cap_capped_routes);
        dst.highfreq_cap_capped_routes = saturating_add_u64(dst.highfreq_cap_capped_routes, src.highfreq_cap_capped_routes);
        dst.manifest_cap_queries = saturating_add_u64(dst.manifest_cap_queries, src.manifest_cap_queries);
        dst.manifest_cap_max = std::max(dst.manifest_cap_max, src.manifest_cap_max);
        dst.original_exact_lowfreq_rows = saturating_add_u64(dst.original_exact_lowfreq_rows, src.original_exact_lowfreq_rows);
        dst.emitted_exact_lowfreq_rows = saturating_add_u64(dst.emitted_exact_lowfreq_rows, src.emitted_exact_lowfreq_rows);
        dst.original_highfreq_cost_proxy = saturating_add_u64(dst.original_highfreq_cost_proxy, src.original_highfreq_cost_proxy);
        dst.emitted_highfreq_cost_proxy = saturating_add_u64(dst.emitted_highfreq_cost_proxy, src.emitted_highfreq_cost_proxy);
        dst.original_exact_label_cardinality_sum = saturating_add_u64(dst.original_exact_label_cardinality_sum, src.original_exact_label_cardinality_sum);
        dst.emitted_exact_label_cardinality_sum = saturating_add_u64(dst.emitted_exact_label_cardinality_sum, src.emitted_exact_label_cardinality_sum);
        dst.sketch_labels = saturating_add_u64(dst.sketch_labels, src.sketch_labels);
        dst.sketch_hashes = saturating_add_u64(dst.sketch_hashes, src.sketch_hashes);
        dst.sketch_bytes = saturating_add_u64(dst.sketch_bytes, src.sketch_bytes);
        dst.sketch_sidecar_bytes_read = saturating_add_u64(dst.sketch_sidecar_bytes_read, src.sketch_sidecar_bytes_read);
        dst.sketch_build_rows = saturating_add_u64(dst.sketch_build_rows, src.sketch_build_rows);
        dst.retained_zero_marginal_routes = saturating_add_u64(dst.retained_zero_marginal_routes, src.retained_zero_marginal_routes);
        dst.exact_cardinality_used = dst.exact_cardinality_used || src.exact_cardinality_used;
        dst.estimated_union_coverage_sum = dst.estimated_union_coverage_sum + src.estimated_union_coverage_sum;
        dst.retained_estimated_marginal_gain_sum = dst.retained_estimated_marginal_gain_sum + src.retained_estimated_marginal_gain_sum;
        dst.original_predicted_latency_us_sum = dst.original_predicted_latency_us_sum + src.original_predicted_latency_us_sum;
        dst.emitted_predicted_latency_us_sum = dst.emitted_predicted_latency_us_sum + src.emitted_predicted_latency_us_sum;
        dst.emitted_quality_sum = dst.emitted_quality_sum + src.emitted_quality_sum;
        dst.emitted_utility_sum = dst.emitted_utility_sum + src.emitted_utility_sum;
    }
    if (route_offset > UINT32_MAX) fail_graph("routed query count exceeds uint32 range");
    runtime.query_vectors.rows = static_cast<uint32_t>(route_offset);
    runtime.query_float_vectors.rows = dense_vector_scalar_is_float(runtime.vector_scalar_type) ? runtime.query_vectors.rows : 0;
    runtime.query_labels_matrix.rows = runtime.query_vectors.rows;
    runtime.query_ranges.rows = runtime.query_vectors.rows;
    report_prepared_query_routes(runtime);
    std::printf("[query-prepare] stage=routes optimized=1 threads=%zu chunks=%zu original_queries=%u routes=%u online=1\n",
                workers, chunk_count, rows, runtime.query_vectors.rows);
}

void validate_graph_runtime_query_metadata(const GraphRuntime& runtime,
                                            const std::string& context)
{
    if (runtime.route_payload_preparation.deferred)
        fail_graph(context + ": execution payloads have not been materialized");
    const size_t routed_query_count = runtime.query_group_ids.size();
    const auto fail_context = [&](const std::string& message) {
        fail_graph(context + ": " + message);
    };

    if (runtime.metadata.label_count == 0) {
        if (routed_query_count != 0 ||
            !runtime.query_filter_labels.empty() ||
            !runtime.query_filter_bits.empty() ||
            !runtime.start_local_ids.empty() ||
            !runtime.query_is_lowfreq.empty() ||
            !runtime.route_to_original_query.empty() ||
            !runtime.route_to_clause.empty() ||
            runtime.route_clause_indptr.size() > 1 ||
            !runtime.route_clause_ids.empty() ||
            !runtime.route_top_branch_ids.empty() ||
            !runtime.route_enabled.empty() ||
            !runtime.route_providers.empty() ||
            !runtime.route_execution_modes.empty() ||
            !runtime.route_candidate_sources.empty() ||
            runtime.query_vectors.rows != 0 ||
            runtime.query_labels_matrix.rows != 0) {
            fail_context("metadata.label_count is 0 but routed query metadata is non-empty");
        }
        return;
    }

    if (runtime.metadata.label_size.size() != runtime.metadata.label_count) {
        fail_context("metadata.label_size size does not match metadata.label_count");
    }
    if (!runtime.original_query_label_ops.empty() &&
        runtime.original_query_label_ops.size() != runtime.original_query_count) {
        fail_context("original_query_label_ops size does not match original_query_count");
    }
    validate_query_occurrence_metadata(runtime.original_query_labels_matrix,
                                       context + ": original query occurrence metadata");
    validate_query_occurrence_metadata(runtime.query_labels_matrix,
                                       context + ": routed query occurrence metadata");
    if (runtime.metadata.label_offset.size() != runtime.metadata.label_count) {
        fail_context("metadata.label_offset size does not match metadata.label_count");
    }
    if (runtime.group_layout.groups.size() != runtime.metadata.label_count) {
        fail_context("group_layout.groups size does not match metadata.label_count");
    }
    if (runtime.query_filter_labels.size() != routed_query_count ||
        runtime.query_filter_bits.size() != routed_query_count ||
        runtime.start_local_ids.size() != routed_query_count ||
        runtime.query_is_lowfreq.size() != routed_query_count ||
        runtime.route_to_original_query.size() != routed_query_count ||
        runtime.route_to_clause.size() != routed_query_count ||
        runtime.route_clause_indptr.size() != routed_query_count + 1 ||
        runtime.route_clause_indptr.back() != runtime.route_clause_ids.size() ||
        runtime.route_top_branch_ids.size() != routed_query_count ||
        runtime.route_enabled.size() != routed_query_count ||
        runtime.route_providers.size() != routed_query_count ||
        runtime.route_execution_modes.size() != routed_query_count ||
        runtime.route_candidate_sources.size() != routed_query_count ||
        runtime.query_ranges.rows != routed_query_count ||
        runtime.query_ranges.modes.size() != routed_query_count * runtime.query_ranges.range_count ||
        runtime.query_ranges.lower_bounds.size() != routed_query_count * runtime.query_ranges.range_count ||
        runtime.query_ranges.upper_bounds.size() != routed_query_count * runtime.query_ranges.range_count) {
        fail_context("routed query metadata vectors are not size-aligned");
    }
    if (runtime.range_access_planner_profile.planned &&
        (runtime.route_range_posting_begins.size() != routed_query_count ||
         runtime.route_range_posting_sizes.size() != routed_query_count ||
         runtime.route_range_posting_indices.size() != routed_query_count ||
         runtime.range_graph_and_original_queries.size() != runtime.original_query_count ||
         runtime.range_graph_or_original_queries.size() != runtime.original_query_count ||
         runtime.range_graph_clause_indptr.size() != runtime.original_query_count + 1 ||
         runtime.range_graph_clause_indptr.back() !=
             runtime.range_graph_clause_ids.size())) {
        fail_context("range access plan is not aligned with routes and original queries");
    }
    if (runtime.query_vectors.rows != routed_query_count ||
        runtime.query_labels_matrix.rows != routed_query_count ||
        runtime.query_labels_matrix.row_labels.size() != routed_query_count) {
        fail_context("routed query matrices are not aligned with query_group_ids");
    }
    if (dense_vector_scalar_is_float(runtime.vector_scalar_type) &&
        (runtime.query_float_vectors.rows != routed_query_count ||
         runtime.query_float_vectors.cols != runtime.query_vectors.cols ||
         runtime.query_float_vectors.data.size() != (runtime.shared_route_query_vectors ? 0 :
             routed_query_count * static_cast<size_t>(runtime.query_float_vectors.cols)))) {
        fail_context("routed float query matrix is not aligned with query_group_ids");
    }

    if (runtime.shared_route_query_vectors) {
        const size_t values = static_cast<size_t>(runtime.original_query_count) * runtime.query_vectors.cols;
        if (!runtime.query_vectors.data.empty() || !runtime.query_float_vectors.data.empty() ||
            (runtime.vector_scalar_type == DenseVectorScalarType::Uint8
                ? runtime.original_query_vectors.data.size() != values
                : runtime.original_query_float_vectors.data.size() != values))
            fail_context("shared route vectors require one complete original query matrix");
    }

    for (size_t route_idx = 0; route_idx < routed_query_count; ++route_idx) {
        if (runtime.route_enabled[route_idx] > 1) {
            fail_context("route_enabled contains a non-boolean value");
        }
        const QueryRouteProvider provider = runtime.route_providers[route_idx];
        const QueryRouteExecution execution = runtime.route_execution_modes[route_idx];
        const QueryCandidateSource candidate_source = runtime.route_candidate_sources[route_idx];
        if ((runtime.supported_query_route_providers & query_route_provider_bit(provider)) == 0) {
            fail_context("route uses a provider not supported by this backend");
        }
        const uint32_t clause_id = runtime.route_to_clause[route_idx];
        if (clause_id != UINT32_MAX &&
            clause_id >= runtime.original_query_clauses.clause_count()) {
            fail_context("route_to_clause contains an out-of-range clause id");
        }
        const uint64_t route_clause_begin = runtime.route_clause_indptr[route_idx];
        const uint64_t route_clause_end = runtime.route_clause_indptr[route_idx + 1];
        if (route_clause_begin > route_clause_end ||
            route_clause_end > runtime.route_clause_ids.size()) {
            fail_context("route clause CSR is invalid");
        }
        for (uint64_t idx = route_clause_begin; idx < route_clause_end; ++idx) {
            if (runtime.route_clause_ids[idx] >=
                runtime.original_query_clauses.clause_count()) {
                fail_context("route clause CSR contains an out-of-range clause id");
            }
        }
        const uint32_t group_id = runtime.query_group_ids[route_idx];
        if (provider == QueryRouteProvider::RangePosting) {
            if (group_id != UINT32_MAX ||
                runtime.query_filter_labels[route_idx] != UINT32_MAX ||
                runtime.query_filter_bits[route_idx] != 0 ||
                candidate_source != QueryCandidateSource::RangePosting ||
                (execution != QueryRouteExecution::GpuResidentScan &&
                 execution != QueryRouteExecution::HostCandidateStream &&
                 execution != QueryRouteExecution::GpuResidentScanPostFilter) ||
                !query_has_any_active_range(runtime.query_ranges,
                                            static_cast<uint32_t>(route_idx))) {
                fail_context("range-posting route has invalid label/range metadata");
            }
            continue;
        }
        if (execution == QueryRouteExecution::HostCandidateStream &&
            candidate_source == QueryCandidateSource::None) {
            fail_context("host candidate stream route has no candidate source");
        }
        if (group_id == UINT32_MAX) {
            continue;
        }
        if (group_id >= runtime.metadata.label_count) {
            fail_context("query_group_ids contains group_id out of range at route " +
                         std::to_string(route_idx) +
                         " (group_id=" + std::to_string(group_id) +
                         ", label_count=" + std::to_string(runtime.metadata.label_count) + ")");
        }
    }
}

static size_t retain_graph_routes(
    GraphRuntime& runtime,
    const std::vector<uint8_t>& selected_original_queries,
    const std::vector<uint8_t>* selected_routes)
{
    if (selected_original_queries.size() != runtime.original_query_count) {
        fail_graph("route selection mask must match original query count");
    }
    validate_graph_runtime_query_metadata(runtime,
                                            "retain_graph_routes_for_original_queries(before)");

    const size_t previous_route_count = runtime.route_to_original_query.size();
    if (selected_routes != nullptr && selected_routes->size() != previous_route_count) {
        fail_graph("route selection mask must match routed query count");
    }
    DenseUint8Matrix selected_query_vectors;
    selected_query_vectors.cols = runtime.query_vectors.cols;
    DenseFloatMatrix selected_query_float_vectors;
    selected_query_float_vectors.cols = runtime.query_float_vectors.cols;
    SparseLabelMatrix selected_query_labels;
    selected_query_labels.cols = runtime.query_labels_matrix.cols;
    selected_query_labels.indptr.push_back(0);
    QueryRangeTable selected_query_ranges;
    selected_query_ranges.range_count = runtime.query_ranges.range_count;

    std::vector<uint32_t> selected_query_group_ids;
    std::vector<uint32_t> selected_query_filter_labels;
    std::vector<uint64_t> selected_query_filter_bits;
    std::vector<uint32_t> selected_start_local_ids;
    std::vector<uint8_t> selected_query_is_lowfreq;
    std::vector<uint32_t> selected_route_to_original_query;
    std::vector<uint32_t> selected_route_to_clause;
    std::vector<uint64_t> selected_route_clause_indptr{0};
    std::vector<uint32_t> selected_route_clause_ids;
    std::vector<uint32_t> selected_route_top_branch_ids;
    std::vector<uint8_t> selected_route_enabled;
    std::vector<QueryRouteProvider> selected_route_providers;
    std::vector<QueryRouteExecution> selected_route_execution_modes;
    std::vector<QueryCandidateSource> selected_route_candidate_sources;
    std::vector<uint64_t> selected_route_range_posting_begins;
    std::vector<uint64_t> selected_route_range_posting_sizes;
    std::vector<uint32_t> selected_route_range_posting_indices;

    selected_query_group_ids.reserve(previous_route_count);
    selected_query_filter_labels.reserve(previous_route_count);
    selected_query_filter_bits.reserve(previous_route_count);
    selected_start_local_ids.reserve(previous_route_count);
    selected_query_is_lowfreq.reserve(previous_route_count);
    selected_route_to_original_query.reserve(previous_route_count);
    selected_route_to_clause.reserve(previous_route_count);
    selected_route_top_branch_ids.reserve(previous_route_count);
    selected_route_enabled.reserve(previous_route_count);
    selected_route_providers.reserve(previous_route_count);
    selected_route_execution_modes.reserve(previous_route_count);
    selected_route_candidate_sources.reserve(previous_route_count);
    selected_route_range_posting_begins.reserve(previous_route_count);
    selected_route_range_posting_sizes.reserve(previous_route_count);
    selected_route_range_posting_indices.reserve(previous_route_count);

    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        selected_query_vectors.data.reserve(runtime.query_vectors.data.size());
    } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        selected_query_float_vectors.data.reserve(runtime.query_float_vectors.data.size());
    } else {
        fail_graph("unsupported dense query scalar type while retaining routes");
    }

    for (size_t route_idx = 0; route_idx < previous_route_count; ++route_idx) {
        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        if (original_query >= selected_original_queries.size()) {
            fail_graph("route-to-original query id exceeds selection mask");
        }
        if (selected_original_queries[original_query] == 0 ||
            (selected_routes != nullptr && (*selected_routes)[route_idx] == 0)) {
            continue;
        }

        if (!runtime.shared_route_query_vectors && runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            const uint8_t* query = runtime.query_vectors.row_ptr(static_cast<uint32_t>(route_idx));
            selected_query_vectors.data.insert(selected_query_vectors.data.end(),
                                               query,
                                               query + runtime.query_vectors.cols);
        } else if (!runtime.shared_route_query_vectors) {
            const float* query =
                runtime.query_float_vectors.row_ptr(static_cast<uint32_t>(route_idx));
            selected_query_float_vectors.data.insert(selected_query_float_vectors.data.end(),
                                                     query,
                                                     query + runtime.query_float_vectors.cols);
        }

        const SparseLabelRowSpan labels = sparse_label_row_span(
            runtime.query_labels_matrix, static_cast<uint32_t>(route_idx));
        if (!labels.valid) {
            fail_graph("routed query label CSR is invalid while retaining routes");
        }
        const uint64_t label_begin = runtime.query_labels_matrix.indptr[route_idx];
        const uint64_t label_end = runtime.query_labels_matrix.indptr[route_idx + 1];
        if (labels.begin != labels.end) {
            selected_query_labels.indices.insert(selected_query_labels.indices.end(),
                                                 labels.begin,
                                                 labels.end);
            if (!runtime.query_labels_matrix.source_tiers.empty()) {
                selected_query_labels.source_tiers.insert(
                    selected_query_labels.source_tiers.end(),
                    runtime.query_labels_matrix.source_tiers.begin() + label_begin,
                    runtime.query_labels_matrix.source_tiers.begin() + label_end);
                selected_query_labels.relevance_ranks.insert(
                    selected_query_labels.relevance_ranks.end(),
                    runtime.query_labels_matrix.relevance_ranks.begin() + label_begin,
                    runtime.query_labels_matrix.relevance_ranks.begin() + label_end);
            }
        }
        if (labels.begin == labels.end) {
            selected_query_labels.row_labels.emplace_back();
        } else {
            selected_query_labels.row_labels.emplace_back(labels.begin, labels.end);
        }
        selected_query_labels.indptr.push_back(selected_query_labels.indices.size());
        ++selected_query_labels.rows;

        const size_t range_base = route_idx * runtime.query_ranges.range_count;
        selected_query_ranges.modes.insert(selected_query_ranges.modes.end(),
                                           runtime.query_ranges.modes.begin() + range_base,
                                           runtime.query_ranges.modes.begin() +
                                               range_base + runtime.query_ranges.range_count);
        selected_query_ranges.lower_bounds.insert(selected_query_ranges.lower_bounds.end(),
                                                  runtime.query_ranges.lower_bounds.begin() + range_base,
                                                  runtime.query_ranges.lower_bounds.begin() +
                                                      range_base + runtime.query_ranges.range_count);
        selected_query_ranges.upper_bounds.insert(selected_query_ranges.upper_bounds.end(),
                                                  runtime.query_ranges.upper_bounds.begin() + range_base,
                                                  runtime.query_ranges.upper_bounds.begin() +
                                                      range_base + runtime.query_ranges.range_count);
        ++selected_query_ranges.rows;

        selected_query_group_ids.push_back(runtime.query_group_ids[route_idx]);
        selected_query_filter_labels.push_back(runtime.query_filter_labels[route_idx]);
        selected_query_filter_bits.push_back(runtime.query_filter_bits[route_idx]);
        selected_start_local_ids.push_back(runtime.start_local_ids[route_idx]);
        selected_query_is_lowfreq.push_back(runtime.query_is_lowfreq[route_idx]);
        selected_route_to_original_query.push_back(original_query);
        selected_route_to_clause.push_back(runtime.route_to_clause[route_idx]);
        const uint64_t route_clause_begin = runtime.route_clause_indptr[route_idx];
        const uint64_t route_clause_end = runtime.route_clause_indptr[route_idx + 1];
        selected_route_clause_ids.insert(
            selected_route_clause_ids.end(),
            runtime.route_clause_ids.begin() + route_clause_begin,
            runtime.route_clause_ids.begin() + route_clause_end);
        selected_route_clause_indptr.push_back(selected_route_clause_ids.size());
        selected_route_top_branch_ids.push_back(runtime.route_top_branch_ids[route_idx]);
        selected_route_enabled.push_back(runtime.route_enabled[route_idx]);
        selected_route_providers.push_back(runtime.route_providers[route_idx]);
        selected_route_execution_modes.push_back(runtime.route_execution_modes[route_idx]);
        selected_route_candidate_sources.push_back(runtime.route_candidate_sources[route_idx]);
        if (runtime.range_access_planner_profile.planned) {
            selected_route_range_posting_begins.push_back(
                runtime.route_range_posting_begins[route_idx]);
            selected_route_range_posting_sizes.push_back(
                runtime.route_range_posting_sizes[route_idx]);
            selected_route_range_posting_indices.push_back(
                runtime.route_range_posting_indices[route_idx]);
        }
    }

    selected_query_vectors.rows = selected_query_labels.rows;
    selected_query_float_vectors.rows = selected_query_labels.rows;
    runtime.query_vectors = std::move(selected_query_vectors);
    runtime.query_float_vectors = std::move(selected_query_float_vectors);
    runtime.query_labels_matrix = std::move(selected_query_labels);
    runtime.query_ranges = std::move(selected_query_ranges);
    runtime.query_group_ids = std::move(selected_query_group_ids);
    runtime.query_filter_labels = std::move(selected_query_filter_labels);
    runtime.query_filter_bits = std::move(selected_query_filter_bits);
    runtime.start_local_ids = std::move(selected_start_local_ids);
    runtime.query_is_lowfreq = std::move(selected_query_is_lowfreq);
    runtime.route_to_original_query = std::move(selected_route_to_original_query);
    runtime.route_to_clause = std::move(selected_route_to_clause);
    runtime.route_clause_indptr = std::move(selected_route_clause_indptr);
    runtime.route_clause_ids = std::move(selected_route_clause_ids);
    runtime.route_top_branch_ids = std::move(selected_route_top_branch_ids);
    runtime.route_enabled = std::move(selected_route_enabled);
    runtime.route_providers = std::move(selected_route_providers);
    runtime.route_execution_modes = std::move(selected_route_execution_modes);
    runtime.route_candidate_sources = std::move(selected_route_candidate_sources);
    if (runtime.range_access_planner_profile.planned) {
        runtime.route_range_posting_begins =
            std::move(selected_route_range_posting_begins);
        runtime.route_range_posting_sizes =
            std::move(selected_route_range_posting_sizes);
        runtime.route_range_posting_indices =
            std::move(selected_route_range_posting_indices);
    }

    runtime.high_query_indices.clear();
    runtime.low_query_indices.clear();
    for (uint32_t route_idx = 0; route_idx < runtime.route_to_original_query.size(); ++route_idx) {
        if (runtime.route_enabled[route_idx] == 0) {
            continue;
        }
        if (runtime.route_execution_modes[route_idx] == QueryRouteExecution::GraphTraversal) {
            runtime.high_query_indices.push_back(route_idx);
        } else if (runtime.query_is_lowfreq[route_idx] != 0) {
            runtime.low_query_indices.push_back(route_idx);
        }
    }
    validate_graph_runtime_query_metadata(runtime,
                                            "retain_graph_routes_for_original_queries(after)");
    return previous_route_count - runtime.route_to_original_query.size();
}

size_t retain_graph_routes_for_original_queries(
    GraphRuntime& runtime,
    const std::vector<uint8_t>& selected_original_queries)
{
    return retain_graph_routes(runtime, selected_original_queries, nullptr);
}

bool sample_matches_route_predicate(const GraphRuntime& runtime,
                                    uint32_t route_idx, uint32_t sample_id)
{
    if (route_idx >= runtime.route_to_original_query.size() ||
        sample_id >= runtime.base_vectors.rows) return false;
    const uint32_t original = runtime.route_to_original_query[route_idx];
    const auto matches_ranges = [&](const QueryRangeTable& table, uint32_t row) {
        if (row >= table.rows || table.range_count != runtime.base_range_count ||
            (static_cast<size_t>(row) + 1) * table.range_count > table.modes.size() ||
            table.lower_bounds.size() != table.modes.size() ||
            table.upper_bounds.size() != table.modes.size()) return false;
        if (!query_has_any_active_range(table, row)) return true;
        if ((static_cast<size_t>(sample_id) + 1) * runtime.base_range_count >
                runtime.base_range_values.size()) return false;
        return go_fann::range_filter::sample_matches_all_active_ranges_host(
            table, row, runtime.base_range_values, runtime.base_range_count, sample_id);
    };
    const auto& clauses = runtime.original_query_clauses;
    if (runtime.route_clause_indptr.size() == runtime.route_to_original_query.size() + 1) {
        const uint64_t begin = runtime.route_clause_indptr[route_idx];
        const uint64_t end = runtime.route_clause_indptr[route_idx + 1];
        if (begin > end || end > runtime.route_clause_ids.size()) return false;
        if (begin != end) {
            if (original >= runtime.original_query_count ||
                clauses.query_indptr.size() != static_cast<size_t>(runtime.original_query_count) + 1)
                return false;
            for (uint64_t i = begin; i < end; ++i) {
                const uint32_t clause = runtime.route_clause_ids[i];
                if (clause < clauses.query_indptr[original] ||
                    clause >= clauses.query_indptr[original + 1] ||
                    clause >= clauses.clause_count()) return false;
                const auto labels = clauses.clause_labels(clause, original);
                if (!labels.valid) return false;
                const bool labels_pass = labels.count == 0 ||
                    sample_matches_clause_labels(runtime, clause, sample_id);
                if (labels_pass &&
                    (!clauses.clause_uses_range(clause, original) ||
                     matches_ranges(runtime.original_query_ranges, original))) return true;
            }
            return false;
        }
    }
    const auto labels = sparse_label_row_span(runtime.query_labels_matrix, route_idx);
    return labels.valid && (labels.begin == labels.end ||
        sample_matches_route_clause_labels(runtime, route_idx, sample_id)) &&
        matches_ranges(runtime.query_ranges, route_idx);
}

void materialize_graph_route_payloads(GraphRuntime& runtime)
{
    auto& preparation = runtime.route_payload_preparation;
    if (!preparation.deferred) return;
    const size_t count = runtime.route_to_original_query.size();
    if (preparation.flags.size() != count || preparation.excluded_labels.size() != count || !runtime.query_vectors.data.empty() ||
        !runtime.query_float_vectors.data.empty() || !runtime.query_labels_matrix.indices.empty() ||
        !runtime.query_labels_matrix.row_labels.empty() || !runtime.query_ranges.modes.empty())
        fail_graph("deferred route payloads are not empty/aligned before final materialization");
    const uint32_t inactive_float_rows = runtime.query_float_vectors.rows;
    runtime.query_vectors.rows = runtime.query_float_vectors.rows = 0;
    runtime.query_labels_matrix.rows = runtime.query_ranges.rows = 0;
    runtime.query_labels_matrix.indptr.assign(1, 0);
    if (!runtime.shared_route_query_vectors) {
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8)
            runtime.query_vectors.data.reserve(count * runtime.query_vectors.cols);
        else
            runtime.query_float_vectors.data.reserve(count * runtime.query_float_vectors.cols);
    }
    runtime.query_labels_matrix.row_labels.reserve(count);
    runtime.query_labels_matrix.indptr.reserve(count + 1);
    runtime.query_ranges.modes.reserve(count * runtime.query_ranges.range_count);
    runtime.query_ranges.lower_bounds.reserve(count * runtime.query_ranges.range_count);
    runtime.query_ranges.upper_bounds.reserve(count * runtime.query_ranges.range_count);
    for (size_t route = 0; route < count; ++route) {
        const uint32_t clause_id = runtime.route_to_clause[route];
        const auto clause = clause_id == UINT32_MAX ? HybridClauseView{} :
            hybrid_clause_view(runtime.original_query_clauses, clause_id, runtime.route_to_original_query[route]);
        append_route_payload(runtime, runtime.route_to_original_query[route],
            preparation.excluded_labels[route],
            static_cast<RouteResidualMode>(preparation.flags[route] & 1U),
            clause_id == UINT32_MAX ? nullptr : &clause, (preparation.flags[route] & 2U) != 0);
    }
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8)
        runtime.query_float_vectors.rows = inactive_float_rows;
    preparation.flags.clear();
    preparation.excluded_labels.clear();
    preparation.deferred = false;
    validate_graph_runtime_query_metadata(runtime, "materialize_graph_route_payloads");
    std::printf("[query-prepare] stage=route_payload_materialize routes=%zu vector_bytes=%zu residual_labels=%zu original_atom_occurrences=%zu original_flat_clause_labels=%zu\n",
                count, runtime.query_vectors.data.size() + runtime.query_float_vectors.data.size() * sizeof(float),
                runtime.query_labels_matrix.indices.size(), runtime.original_query_clauses.atom_labels.size(),
                runtime.original_query_clauses.labels.size());
}

// Compact descriptor arrays only. No execution vector, range or residual-label
// buffer exists yet; disabled provider alternatives keep their metadata slots.
static void retain_deferred_route_descriptors(GraphRuntime& runtime,
                                               const std::vector<uint8_t>& retained)
{
    const auto compact = [&](auto& values) {
        if (values.size() != retained.size()) fail_graph("deferred route descriptor size mismatch");
        size_t out = 0;
        for (size_t i = 0; i < retained.size(); ++i)
            if (retained[i]) values[out++] = std::move(values[i]);
        values.resize(out);
    };
    std::vector<uint64_t> offsets{0};
    std::vector<uint32_t> clauses;
    for (size_t i = 0; i < retained.size(); ++i) {
        if (!retained[i]) continue;
        clauses.insert(clauses.end(), runtime.route_clause_ids.begin() + runtime.route_clause_indptr[i],
                       runtime.route_clause_ids.begin() + runtime.route_clause_indptr[i + 1]);
        offsets.push_back(clauses.size());
    }
    runtime.route_clause_ids = std::move(clauses);
    runtime.route_clause_indptr = std::move(offsets);
    compact(runtime.route_to_original_query);
    compact(runtime.route_to_clause);
    compact(runtime.route_top_branch_ids);
    compact(runtime.route_enabled);
    compact(runtime.route_providers);
    compact(runtime.route_execution_modes);
    compact(runtime.route_candidate_sources);
    compact(runtime.query_group_ids);
    compact(runtime.query_filter_labels);
    compact(runtime.query_filter_bits);
    compact(runtime.query_is_lowfreq);
    compact(runtime.start_local_ids);
    compact(runtime.route_payload_preparation.flags);
    compact(runtime.route_payload_preparation.excluded_labels);
    if (runtime.range_access_planner_profile.planned) {
        compact(runtime.route_range_posting_begins);
        compact(runtime.route_range_posting_sizes);
        compact(runtime.route_range_posting_indices);
    }
    const uint32_t count = static_cast<uint32_t>(runtime.route_to_original_query.size());
    runtime.query_vectors.rows = runtime.query_labels_matrix.rows = runtime.query_ranges.rows = count;
    runtime.query_float_vectors.rows = count;
    runtime.high_query_indices.clear();
    runtime.low_query_indices.clear();
    for (uint32_t route = 0; route < count; ++route) {
        if (!runtime.route_enabled[route]) continue;
        if (runtime.route_execution_modes[route] == QueryRouteExecution::GraphTraversal)
            runtime.high_query_indices.push_back(route);
        else if (runtime.query_is_lowfreq[route]) runtime.low_query_indices.push_back(route);
    }
}

static size_t coalesce_graph_routes_legacy(GraphRuntime& runtime)
{
    if (!runtime.route_payload_preparation.deferred)
        validate_graph_runtime_query_metadata(runtime, "coalesce_graph_routes(before)");
    const size_t count = runtime.route_to_original_query.size();
    // Runtime identity scopes this map to one physical index/sample domain.
    // Seeds are execution choices, not separate structures: retain the first
    // selected seed deterministically and search the shared graph once.
    using Key = std::tuple<uint32_t, uint32_t, QueryRouteExecution,
                           QueryCandidateSource, uint8_t>;
    std::map<Key, size_t> representatives;
    std::vector<std::vector<uint32_t>> ownership(count);
    // Disabled provider alternatives carry range-graph mapping/cost information.
    // Only remove duplicate slots created by this pass.
    std::vector<uint8_t> retained(count, 1);
    size_t merged = 0, active = 0;
    for (size_t route = 0; route < count; ++route) {
        const auto begin = runtime.route_clause_indptr[route];
        const auto end = runtime.route_clause_indptr[route + 1];
        ownership[route].assign(runtime.route_clause_ids.begin() + begin,
                                runtime.route_clause_ids.begin() + end);
        if (!runtime.route_enabled[route]) continue;
        ++active;
        if (ownership[route].empty() ||
            runtime.route_providers[route] != QueryRouteProvider::Label ||
            runtime.query_group_ids[route] == UINT32_MAX ||
            runtime.route_execution_modes[route] == QueryRouteExecution::None) continue;
        const uint32_t original = runtime.route_to_original_query[route];
        const auto& clauses = runtime.original_query_clauses;
        if (original >= runtime.original_query_count ||
            clauses.query_indptr.size() != static_cast<size_t>(runtime.original_query_count) + 1)
            fail_graph("route union has invalid original query domain");
        for (uint32_t clause : ownership[route]) {
            if (clause < clauses.query_indptr[original] ||
                clause >= clauses.query_indptr[original + 1] ||
                clause >= clauses.clause_count())
                fail_graph("route union clause belongs to another original query");
        }
        const Key key{original, runtime.query_group_ids[route],
            runtime.route_execution_modes[route], runtime.route_candidate_sources[route],
            runtime.query_is_lowfreq[route]};
        const auto [it, inserted] = representatives.emplace(key, route);
        if (inserted) continue;
        const size_t representative = it->second;
        ownership[representative].insert(ownership[representative].end(),
                                         ownership[route].begin(), ownership[route].end());
        runtime.query_filter_bits[representative] |= runtime.query_filter_bits[route];
        runtime.route_enabled[route] = 0;
        retained[route] = 0;
        ++merged;
    }
    if (merged) {
        runtime.route_clause_indptr.assign(1, 0);
        runtime.route_clause_ids.clear();
        for (size_t route = 0; route < count; ++route) {
            auto& clauses = ownership[route];
            std::sort(clauses.begin(), clauses.end());
            clauses.erase(std::unique(clauses.begin(), clauses.end()), clauses.end());
            runtime.route_clause_ids.insert(runtime.route_clause_ids.end(), clauses.begin(), clauses.end());
            runtime.route_clause_indptr.push_back(runtime.route_clause_ids.size());
            if (runtime.route_enabled[route] && clauses.size() > 1 &&
                runtime.route_providers[route] == QueryRouteProvider::Label) {
                // Flat ranges can only be a necessary prefilter for every
                // disjunct. Exact comparisons remain in the owned DNF.
                const bool all_use_range = std::all_of(clauses.begin(), clauses.end(),
                    [&](uint32_t clause) { return runtime.original_query_clauses.clause_uses_range(clause, runtime.route_to_original_query[route]); });
                if (!all_use_range) {
                    if (runtime.route_payload_preparation.deferred) {
                        runtime.route_payload_preparation.flags[route] |= 2U;
                    } else {
                        const size_t start = route * runtime.query_ranges.range_count;
                        std::fill_n(runtime.query_ranges.modes.begin() + start,
                                    runtime.query_ranges.range_count, uint8_t{0});
                    }
                }
            }
        }
    }
    if (merged) {
        if (runtime.route_payload_preparation.deferred)
            retain_deferred_route_descriptors(runtime, retained);
        else
            retain_graph_routes(runtime, std::vector<uint8_t>(runtime.original_query_count, 1), &retained);
    }
    runtime.or_route_planner_profile.same_producer_merged_routes += merged;
    std::printf("[query-prepare] stage=physical_route_union input_active=%zu merged=%zu output_routes=%u compacted_slots=%zu\n",
                active, merged, runtime.query_vectors.rows, count - runtime.route_to_original_query.size());
    return merged;
}

// Routes remain in their original order. Only the small set of producers owned
// by one query is sorted; clause payloads are flattened after representatives
// have been chosen. No owning container is allocated for every batch route.
size_t coalesce_graph_routes(GraphRuntime& runtime)
{
    const auto config = snapshot_query_preparation_config();
    if (!config.optimized) return coalesce_graph_routes_legacy(runtime);
    if (!runtime.route_payload_preparation.deferred)
        validate_graph_runtime_query_metadata(runtime, "coalesce_graph_routes(before)");
    const size_t count = runtime.route_to_original_query.size();
    const size_t queries = runtime.original_query_count;
    if (runtime.route_clause_indptr.size() != count + 1 ||
        runtime.route_clause_indptr.back() != runtime.route_clause_ids.size())
        fail_graph("route union requires aligned clause CSR");

    std::vector<size_t> query_offsets(queries + 1, 0);
    for (uint32_t query : runtime.route_to_original_query) {
        if (query >= queries) fail_graph("route union has invalid original query domain");
        ++query_offsets[query + 1];
    }
    std::partial_sum(query_offsets.begin(), query_offsets.end(), query_offsets.begin());
    auto cursor = query_offsets;
    std::vector<size_t> query_routes(count), representative(count);
    for (size_t route = 0; route < count; ++route) {
        query_routes[cursor[runtime.route_to_original_query[route]]++] = route;
        representative[route] = route;
    }
    const auto key = [&](size_t route) {
        return std::make_tuple(runtime.query_group_ids[route],
            runtime.route_execution_modes[route], runtime.route_candidate_sources[route],
            runtime.query_is_lowfreq[route]);
    };
    const size_t workers = go_fann::query_prepare::worker_count(config.threads, queries);
    const size_t chunks = std::min(queries, workers * 4);
    std::vector<size_t> active_counts(chunks, 0), merged_counts(chunks, 0);
    go_fann::query_prepare::parallel_tasks(chunks, static_cast<uint32_t>(workers), [&](size_t chunk) {
        std::vector<size_t> producers;
        for (size_t query = queries * chunk / chunks; query < queries * (chunk + 1) / chunks; ++query) {
            producers.clear();
            for (size_t i = query_offsets[query]; i < query_offsets[query + 1]; ++i) {
                const size_t route = query_routes[i];
                if (!runtime.route_enabled[route]) continue;
                ++active_counts[chunk];
                const uint64_t begin = runtime.route_clause_indptr[route];
                const uint64_t end = runtime.route_clause_indptr[route + 1];
                if (begin > end || end > runtime.route_clause_ids.size())
                    fail_graph("route union has invalid clause span");
                if (begin == end || runtime.route_providers[route] != QueryRouteProvider::Label ||
                    runtime.query_group_ids[route] == UINT32_MAX ||
                    runtime.route_execution_modes[route] == QueryRouteExecution::None) continue;
                const auto& clauses = runtime.original_query_clauses;
                if (clauses.query_indptr.size() != queries + 1)
                    fail_graph("route union has invalid clause domain");
                for (uint64_t j = begin; j < end; ++j) {
                    const uint32_t clause = runtime.route_clause_ids[j];
                    if (clause < clauses.query_indptr[query] || clause >= clauses.query_indptr[query + 1] ||
                        clause >= clauses.clause_count())
                        fail_graph("route union clause belongs to another original query");
                }
                producers.push_back(route);
            }
            std::sort(producers.begin(), producers.end(), [&](size_t a, size_t b) {
                return key(a) == key(b) ? a < b : key(a) < key(b);
            });
            for (size_t i = 0; i < producers.size();) {
                const size_t first = producers[i++];
                while (i < producers.size() && key(first) == key(producers[i])) {
                    const size_t duplicate = producers[i++];
                    representative[duplicate] = first;
                    runtime.query_filter_bits[first] |= runtime.query_filter_bits[duplicate];
                    runtime.route_enabled[duplicate] = 0;
                    ++merged_counts[chunk];
                }
            }
        }
    });
    const size_t merged = std::accumulate(merged_counts.begin(), merged_counts.end(), size_t{0});
    const size_t active = std::accumulate(active_counts.begin(), active_counts.end(), size_t{0});
    if (merged) {
        std::vector<uint8_t> retained(count, 1);
        std::vector<size_t> head(count, SIZE_MAX), next(count, SIZE_MAX);
        for (size_t route = count; route-- > 0;) {
            const size_t root = representative[route];
            next[route] = head[root];
            head[root] = route;
            retained[route] = root == route;
        }
        const auto old_offsets = std::move(runtime.route_clause_indptr);
        const auto old_clauses = std::move(runtime.route_clause_ids);
        runtime.route_clause_indptr = {0};
        runtime.route_clause_indptr.reserve(count + 1);
        runtime.route_clause_ids.clear();
        runtime.route_clause_ids.reserve(old_clauses.size());
        std::vector<uint32_t> owned;
        for (size_t route = 0; route < count; ++route) {
            owned.clear();
            for (size_t source = head[route]; source != SIZE_MAX; source = next[source])
                owned.insert(owned.end(), old_clauses.begin() + old_offsets[source],
                             old_clauses.begin() + old_offsets[source + 1]);
            std::sort(owned.begin(), owned.end());
            owned.erase(std::unique(owned.begin(), owned.end()), owned.end());
            runtime.route_clause_ids.insert(runtime.route_clause_ids.end(), owned.begin(), owned.end());
            runtime.route_clause_indptr.push_back(runtime.route_clause_ids.size());
            if (runtime.route_enabled[route] && owned.size() > 1 &&
                runtime.route_providers[route] == QueryRouteProvider::Label &&
                !std::all_of(owned.begin(), owned.end(), [&](uint32_t clause) {
                    return runtime.original_query_clauses.clause_uses_range(clause, runtime.route_to_original_query[route]);
                })) {
                if (runtime.route_payload_preparation.deferred) runtime.route_payload_preparation.flags[route] |= 2U;
                else std::fill_n(runtime.query_ranges.modes.begin() + route * runtime.query_ranges.range_count,
                                 runtime.query_ranges.range_count, uint8_t{0});
            }
        }
        if (runtime.route_payload_preparation.deferred) retain_deferred_route_descriptors(runtime, retained);
        else retain_graph_routes(runtime, std::vector<uint8_t>(queries, 1), &retained);
    }
    runtime.or_route_planner_profile.same_producer_merged_routes += merged;
    std::printf("[query-prepare] stage=physical_route_union input_active=%zu merged=%zu output_routes=%u compacted_slots=%zu query_local=1\n",
                active, merged, runtime.query_vectors.rows, count - runtime.route_to_original_query.size());
    return merged;
}

void plan_graph_range_access(
    GraphRuntime& runtime,
    const go_fann::range_posting::Index& posting_index,
    const RangeAccessPlannerConfig& config)
{
    const auto posting_mode = go_fann::range_posting::parse_mode(config.mode);
    if (posting_mode == go_fann::range_posting::Mode::Off) {
        fail_graph("range access planner requires an enabled range posting mode");
    }
    if (config.and_filter_mode != "auto" && config.and_filter_mode != "pre" &&
        config.and_filter_mode != "post" && config.and_filter_mode != "label") {
        fail_graph(
            "range posting AND filter mode must be one of: auto, pre, post, label");
    }
    if (config.range_label_and_policy != "auto" &&
        config.range_label_and_policy != "label_prefer" &&
        config.range_label_and_policy != "range_prefer" &&
        config.range_label_and_policy != "dual") {
        fail_graph(
            "range label AND policy must be one of: auto, label_prefer, range_prefer, dual");
    }
    if (config.and_filter_mode == "label" &&
        (config.range_label_and_policy == "range_prefer" ||
         config.range_label_and_policy == "dual")) {
        fail_graph(
            "range label AND policy range_prefer or dual cannot use AND filter mode label");
    }
    if (config.scan_threshold == 0) {
        fail_graph("range access planner requires a positive scan threshold");
    }
    if (posting_index.row_count() != runtime.base_vectors.rows ||
        posting_index.range_count() != runtime.base_range_count ||
        posting_index.range_count() != runtime.query_ranges.range_count) {
        fail_graph("range posting index shape does not match the graph runtime runtime");
    }

    const size_t route_count = runtime.route_to_original_query.size();
    runtime.route_range_posting_begins.assign(route_count, 0);
    runtime.route_range_posting_sizes.assign(route_count, 0);
    runtime.route_range_posting_indices.assign(route_count, UINT32_MAX);
    runtime.range_graph_and_original_queries.assign(runtime.original_query_count, 0);
    runtime.range_graph_or_original_queries.assign(runtime.original_query_count, 0);
    runtime.range_access_planner_profile = RangeAccessPlannerProfile{};
    runtime.range_access_planner_profile.planned = true;

    std::vector<std::vector<uint32_t>> route_clauses(route_count);
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (runtime.route_clause_indptr.size() == route_count + 1) {
            const uint64_t begin = runtime.route_clause_indptr[route_idx];
            const uint64_t end = runtime.route_clause_indptr[route_idx + 1];
            if (begin > end || end > runtime.route_clause_ids.size()) {
                fail_graph("range access planner received an invalid route clause CSR");
            }
            route_clauses[route_idx].assign(
                runtime.route_clause_ids.begin() + begin,
                runtime.route_clause_ids.begin() + end);
        } else if (runtime.route_to_clause[route_idx] != UINT32_MAX) {
            route_clauses[route_idx].push_back(runtime.route_to_clause[route_idx]);
        }
    }
    std::vector<std::vector<uint32_t>> range_graph_clauses(
        runtime.original_query_count);

    const size_t clause_count = runtime.original_query_clauses.clause_count();
    // Only retained clause IDs need provider costs. A fuzzy batch may have
    // millions of virtual clauses but only a small capped set of physical routes.
    std::vector<uint32_t> cost_clause_ids;
    for (uint32_t clause : runtime.route_to_clause)
        if (clause != UINT32_MAX) cost_clause_ids.push_back(clause);
    std::sort(cost_clause_ids.begin(), cost_clause_ids.end());
    cost_clause_ids.erase(std::unique(cost_clause_ids.begin(), cost_clause_ids.end()), cost_clause_ids.end());
    const auto cost_slot = [&](uint32_t clause) -> size_t {
        const auto it = std::lower_bound(cost_clause_ids.begin(), cost_clause_ids.end(), clause);
        if (it == cost_clause_ids.end() || *it != clause) fail_graph("range provider cost has no retained clause");
        return static_cast<size_t>(it - cost_clause_ids.begin());
    };
    std::vector<uint64_t> clause_best_label_rows(cost_clause_ids.size(), UINT64_MAX);
    std::vector<uint8_t> clause_has_lowfreq(cost_clause_ids.size(), 0);
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (runtime.route_providers[route_idx] != QueryRouteProvider::Label ||
            runtime.route_to_clause[route_idx] == UINT32_MAX) {
            continue;
        }
        const uint32_t clause = runtime.route_to_clause[route_idx];
        if (clause >= clause_count) {
            fail_graph("range access planner received an invalid label clause id");
        }
        const uint32_t group = runtime.query_group_ids[route_idx];
        uint64_t rows = UINT64_MAX;
        if (runtime.query_is_lowfreq[route_idx] != 0 &&
            group < runtime.lowfreq_index.label_size.size()) {
            rows = runtime.lowfreq_index.label_size[group];
            clause_has_lowfreq[cost_slot(clause)] = 1;
        } else if (group < runtime.metadata.group_posting_size.size()) {
            rows = runtime.metadata.group_posting_size[group];
        }
        clause_best_label_rows[cost_slot(clause)] = std::min(clause_best_label_rows[cost_slot(clause)], rows);
    }

    // Built online for this batch. Query-local CSR avoids scanning every
    // route whenever one clause or pure-range query selects a provider.
    const bool indexed_routes = snapshot_query_preparation_config().optimized;
    std::vector<size_t> query_route_offsets, query_route_ids;
    if (indexed_routes) {
        query_route_offsets.assign(static_cast<size_t>(runtime.original_query_count) + 1, 0);
        for (uint32_t query : runtime.route_to_original_query) {
            if (query >= runtime.original_query_count) fail_graph("invalid original query in range planner");
            ++query_route_offsets[query + 1];
        }
        std::partial_sum(query_route_offsets.begin(), query_route_offsets.end(), query_route_offsets.begin());
        auto cursor = query_route_offsets;
        query_route_ids.resize(route_count);
        for (size_t route = 0; route < route_count; ++route)
            query_route_ids[cursor[runtime.route_to_original_query[route]]++] = route;
    }
    const auto visit_query_routes = [&](uint32_t query, const auto& visit) {
        if (indexed_routes) {
            for (size_t i = query_route_offsets[query]; i < query_route_offsets[query + 1]; ++i)
                visit(query_route_ids[i]);
        } else {
            for (size_t route = 0; route < route_count; ++route) visit(route);
        }
    };
    const auto disable_clause_label_routes = [&](uint32_t clause, bool keep_lowfreq, uint32_t query) {
        visit_query_routes(query, [&](size_t other) {
            if (runtime.route_providers[other] == QueryRouteProvider::Label &&
                runtime.route_to_clause[other] == clause &&
                !(keep_lowfreq && runtime.query_is_lowfreq[other] != 0))
                runtime.route_enabled[other] = 0;
        });
    };
    const auto disable_pure_range_graph_routes = [&](uint32_t query) {
        visit_query_routes(query, [&](size_t other) {
            if (runtime.route_providers[other] == QueryRouteProvider::Label &&
                runtime.route_to_original_query[other] == query)
                runtime.route_enabled[other] = 0;
        });
    };

    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (runtime.route_providers[route_idx] != QueryRouteProvider::RangePosting) {
            continue;
        }
        const auto slice = go_fann::range_posting::choose_smallest_slice(
            posting_index,
            runtime.route_payload_preparation.deferred ? runtime.original_query_ranges : runtime.query_ranges,
            runtime.route_payload_preparation.deferred ? runtime.route_to_original_query[route_idx]
                                                       : static_cast<uint32_t>(route_idx));
        if (!slice.has_active_range || slice.range_index == UINT32_MAX) {
            fail_graph("range-posting route has no active range slice");
        }
        runtime.route_range_posting_begins[route_idx] = slice.begin;
        runtime.route_range_posting_sizes[route_idx] = slice.count;
        runtime.route_range_posting_indices[route_idx] = slice.range_index;

        const uint32_t original_query = runtime.route_to_original_query[route_idx];
        const uint32_t clause = runtime.route_to_clause[route_idx];
        bool has_residual_labels;
        if (runtime.route_payload_preparation.deferred) {
            if (clause != UINT32_MAX) {
                const auto labels = runtime.original_query_clauses.clause_labels(clause, original_query);
                if (!labels.valid) fail_graph("range-posting route has invalid clause labels");
                has_residual_labels = labels.count != 0;
            } else {
                const auto labels = sparse_label_row_span(runtime.original_query_labels_matrix, original_query);
                if (!labels.valid) fail_graph("range-posting route has invalid original labels");
                has_residual_labels = labels.begin != labels.end;
            }
            has_residual_labels = has_residual_labels && !(runtime.route_payload_preparation.flags[route_idx] & 1U);
        } else {
            const auto labels = sparse_label_row_span(runtime.query_labels_matrix, static_cast<uint32_t>(route_idx));
            if (!labels.valid) fail_graph("range-posting route has invalid residual-label metadata");
            has_residual_labels = labels.begin != labels.end;
        }
        const bool eligible = slice.count <= config.scan_threshold;

        if (!has_residual_labels) {
            const bool hybrid_or =
                original_query < runtime.original_query_label_ops.size() &&
                runtime.original_query_clauses.has_clauses(original_query) &&
                runtime.original_query_label_ops[original_query] == QueryLabelOp::Or;
            if (!eligible && hybrid_or && config.range_graph_hybrid_or_enabled) {
                runtime.route_enabled[route_idx] = 0;
                runtime.range_graph_or_original_queries[original_query] = 1;
                if (clause != UINT32_MAX) {
                    range_graph_clauses[original_query].push_back(clause);
                }
                continue;
            }
            if (!eligible && hybrid_or) {
                fail_graph(
                    "wide hybrid range-OR route exceeds range posting threshold "
                    "without an enabled range-graph fallback");
            }
            if (!eligible) {
                runtime.route_enabled[route_idx] = 0;
                continue;
            }
            runtime.route_execution_modes[route_idx] = QueryRouteExecution::GpuResidentScan;
            if (!runtime.original_query_clauses.has_clauses(original_query) &&
                posting_mode == go_fann::range_posting::Mode::Select) {
                disable_pure_range_graph_routes(original_query);
            }
        } else {
            if (clause == UINT32_MAX || clause >= clause_count) {
                fail_graph("range-label conjunction has no valid DNF clause id");
            }
            const bool auto_policy = config.range_label_and_policy == "auto";
            const bool label_prefer =
                config.range_label_and_policy == "label_prefer" ||
                (auto_policy && config.and_filter_mode == "label");
            const bool range_prefer = config.range_label_and_policy == "range_prefer";
            const bool dual = config.range_label_and_policy == "dual";
            if (auto_policy && clause_has_lowfreq[cost_slot(clause)] != 0) {
                runtime.route_enabled[route_idx] = 0;
                ++runtime.range_access_planner_profile.lowfreq_priority_clauses;
                continue;
            }
            if (label_prefer) {
                runtime.route_enabled[route_idx] = 0;
                ++runtime.range_access_planner_profile.label_selected_clauses;
                continue;
            }

            const bool select_post =
                (config.and_filter_mode == "auto" || config.and_filter_mode == "post") &&
                eligible;
            const bool select_pre =
                config.and_filter_mode == "pre" && eligible &&
                (!auto_policy || slice.count < clause_best_label_rows[cost_slot(clause)]);
            const bool select_range_graph =
                !select_post && !select_pre &&
                config.range_graph_hybrid_and_enabled &&
                (!auto_policy || slice.count < clause_best_label_rows[cost_slot(clause)]);

            if (select_post) {
                runtime.route_execution_modes[route_idx] =
                    QueryRouteExecution::GpuResidentScanPostFilter;
                ++runtime.range_access_planner_profile.postfilter_routes;
                if (!dual &&
                    posting_mode == go_fann::range_posting::Mode::Select) {
                    disable_clause_label_routes(clause, false, original_query);
                }
            } else if (select_pre) {
                runtime.route_execution_modes[route_idx] =
                    QueryRouteExecution::HostCandidateStream;
                ++runtime.range_access_planner_profile.prefilter_routes;
                if (!dual &&
                    posting_mode == go_fann::range_posting::Mode::Select) {
                    disable_clause_label_routes(clause, false, original_query);
                }
            } else if (select_range_graph) {
                runtime.route_enabled[route_idx] = 0;
                runtime.range_graph_and_original_queries[original_query] = 1;
                range_graph_clauses[original_query].push_back(clause);
                if (!dual) {
                    disable_clause_label_routes(clause, false, original_query);
                }
            } else if (range_prefer || dual) {
                fail_graph(
                    "forced range label AND policy has no threshold-safe range provider");
            } else {
                runtime.route_enabled[route_idx] = 0;
                ++runtime.range_access_planner_profile.label_selected_clauses;
            }
            if (dual && (select_post || select_pre || select_range_graph)) {
                ++runtime.range_access_planner_profile.dual_selected_clauses;
            }
        }

    }

    struct RangeRouteKey {
        uint32_t original_query = UINT32_MAX;
        uint32_t range_index = UINT32_MAX;
        uint64_t begin = 0;
        uint64_t size = 0;
        QueryRouteExecution execution = QueryRouteExecution::None;

        bool operator==(const RangeRouteKey& other) const
        {
            return original_query == other.original_query &&
                   range_index == other.range_index && begin == other.begin &&
                   size == other.size && execution == other.execution;
        }
    };
    struct RangeRouteKeyHash {
        size_t operator()(const RangeRouteKey& key) const
        {
            size_t hash = std::hash<uint64_t>{}(key.begin);
            const auto mix = [&](uint64_t value) {
                hash ^= std::hash<uint64_t>{}(value) + 0x9e3779b97f4a7c15ULL +
                        (hash << 6U) + (hash >> 2U);
            };
            mix(key.size);
            mix(key.original_query);
            mix(key.range_index);
            mix(static_cast<uint8_t>(key.execution));
            return hash;
        }
    };
    std::unordered_map<RangeRouteKey, size_t, RangeRouteKeyHash> representative_by_key;
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (runtime.route_enabled[route_idx] == 0 ||
            runtime.route_providers[route_idx] != QueryRouteProvider::RangePosting) {
            continue;
        }
        const RangeRouteKey key{
            runtime.route_to_original_query[route_idx],
            runtime.route_range_posting_indices[route_idx],
            runtime.route_range_posting_begins[route_idx],
            runtime.route_range_posting_sizes[route_idx],
            runtime.route_execution_modes[route_idx]};
        const auto [it, inserted] = representative_by_key.emplace(key, route_idx);
        if (!inserted) {
            std::vector<uint32_t>& representative_clauses = route_clauses[it->second];
            representative_clauses.insert(representative_clauses.end(),
                                          route_clauses[route_idx].begin(),
                                          route_clauses[route_idx].end());
            runtime.route_enabled[route_idx] = 0;
        }
    }

    runtime.route_clause_indptr.assign(1, 0);
    runtime.route_clause_ids.clear();
    for (std::vector<uint32_t>& clauses : route_clauses) {
        std::sort(clauses.begin(), clauses.end());
        clauses.erase(std::unique(clauses.begin(), clauses.end()), clauses.end());
        runtime.route_clause_ids.insert(runtime.route_clause_ids.end(),
                                        clauses.begin(), clauses.end());
        runtime.route_clause_indptr.push_back(runtime.route_clause_ids.size());
    }
    runtime.range_graph_clause_indptr.assign(1, 0);
    runtime.range_graph_clause_ids.clear();
    for (std::vector<uint32_t>& clauses : range_graph_clauses) {
        std::sort(clauses.begin(), clauses.end());
        clauses.erase(std::unique(clauses.begin(), clauses.end()), clauses.end());
        runtime.range_graph_clause_ids.insert(runtime.range_graph_clause_ids.end(),
                                              clauses.begin(), clauses.end());
        runtime.range_graph_clause_indptr.push_back(
            runtime.range_graph_clause_ids.size());
    }

    runtime.range_access_planner_profile.range_posting_routes = 0;
    runtime.range_access_planner_profile.range_posting_rows = 0;
    runtime.range_access_planner_profile.postfilter_routes = 0;
    runtime.range_access_planner_profile.prefilter_routes = 0;
    for (size_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (runtime.route_enabled[route_idx] == 0 ||
            runtime.route_providers[route_idx] != QueryRouteProvider::RangePosting) {
            continue;
        }
        ++runtime.range_access_planner_profile.range_posting_routes;
        runtime.range_access_planner_profile.range_posting_rows +=
            runtime.route_range_posting_sizes[route_idx];
        if (runtime.route_execution_modes[route_idx] ==
            QueryRouteExecution::GpuResidentScanPostFilter) {
            ++runtime.range_access_planner_profile.postfilter_routes;
        } else if (runtime.route_execution_modes[route_idx] ==
                   QueryRouteExecution::HostCandidateStream) {
            ++runtime.range_access_planner_profile.prefilter_routes;
        }
    }

    runtime.range_access_planner_profile.range_graph_and_queries =
        std::count(runtime.range_graph_and_original_queries.begin(),
                   runtime.range_graph_and_original_queries.end(), uint8_t{1});
    runtime.range_access_planner_profile.range_graph_or_queries =
        std::count(runtime.range_graph_or_original_queries.begin(),
                   runtime.range_graph_or_original_queries.end(), uint8_t{1});
    runtime.high_query_indices.clear();
    runtime.low_query_indices.clear();
    for (uint32_t route_idx = 0; route_idx < route_count; ++route_idx) {
        if (runtime.route_enabled[route_idx] == 0) continue;
        if (runtime.route_execution_modes[route_idx] == QueryRouteExecution::GraphTraversal) {
            runtime.high_query_indices.push_back(route_idx);
        } else if (runtime.query_is_lowfreq[route_idx] != 0) {
            runtime.low_query_indices.push_back(route_idx);
        }
    }

    std::printf(
        "[runtime-range-access] routes=%llu rows=%llu post=%llu pre=%llu "
        "policy=%s lowfreq_priority=%llu label=%llu dual=%llu "
        "range_graph_and=%llu range_graph_or=%llu\n",
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.range_posting_routes),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.range_posting_rows),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.postfilter_routes),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.prefilter_routes),
        config.range_label_and_policy.c_str(),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.lowfreq_priority_clauses),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.label_selected_clauses),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.dual_selected_clauses),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.range_graph_and_queries),
        static_cast<unsigned long long>(
            runtime.range_access_planner_profile.range_graph_or_queries));
}

void ensure_host_candidate_label_data(GraphRuntime& runtime,
                                      const GraphLoadConfig& config)
{
    bool requires_host_candidate_labels = false;
    for (size_t route_idx = 0; route_idx < runtime.route_execution_modes.size(); ++route_idx) {
        if (runtime.route_execution_modes[route_idx] !=
                QueryRouteExecution::HostCandidateStream &&
            runtime.route_execution_modes[route_idx] !=
                QueryRouteExecution::GpuResidentScanPostFilter) {
            continue;
        }
        const SparseLabelRowSpan labels = sparse_label_row_span(
            runtime.query_labels_matrix, static_cast<uint32_t>(route_idx));
        if (!labels.valid) {
            fail_graph("host candidate stream route has invalid residual-label metadata");
        }
        requires_host_candidate_labels = labels.begin != labels.end;
        if (requires_host_candidate_labels) break;
    }
    if (!requires_host_candidate_labels) return;

    const size_t expected_indptr_size = static_cast<size_t>(runtime.base_vectors.rows) + 1;
    if (runtime.data_labels.indptr.size() == expected_indptr_size) return;
    runtime.data_labels = load_spmat_labels(config.data_label_spmat_path,
                                            config.dataset_cutoff);
    if (runtime.data_labels.rows != runtime.base_vectors.rows ||
        runtime.data_labels.indptr.size() != expected_indptr_size) {
        fail_graph("host candidate streaming requires base-row label CSR data");
    }
}

GraphRuntime load_graph_runtime(const GraphLoadConfig& config)
{
    if (config.graph_storage_mode != "memory" && config.graph_storage_mode != "default") {
        fail_graph("graph_storage_mode must be memory or default; disk runtime is unsupported");
    }
    const auto total_start = LoadClock::now();
    auto stage_start = total_start;
    GraphRuntime runtime;
    runtime.shared_route_query_vectors = config.supports_shared_query_vectors && snapshot_query_preparation_config().optimized;
    runtime.rerank_vector_storage_mode = config.rerank_vector_storage_mode;
    runtime.rerank_vector_artifact_path = config.rerank_vector_artifact_path;
    if (!go_fann::defaults::is_valid_rerank_vector_storage_mode(
            runtime.rerank_vector_storage_mode)) {
        fail_graph("rerank_vector_storage_mode must be one of: default, mmap_fp16, memory_int8");
    }
    runtime.and_dual_route_posting_ratio_threshold =
        config.and_dual_route_posting_ratio_threshold;
    runtime.supported_query_route_providers = config.supported_query_route_providers;
    runtime.base_vector_metadata = inspect_dense_vector_file(config.base_vector_path,
                                                             config.dataset_cutoff);
    const auto query_vector_inspect_start = LoadClock::now();
    runtime.query_vector_metadata = inspect_dense_vector_file(config.query_vector_path);
    runtime.query_preparation_timing.query_vector_inspect_ms =
        ns_to_ms(elapsed_load_ns(query_vector_inspect_start));
    if (runtime.base_vector_metadata.scalar_type != runtime.query_vector_metadata.scalar_type) {
        fail_graph("base/query vector scalar type mismatch: base=" +
                     std::string(dense_vector_scalar_type_name(runtime.base_vector_metadata.scalar_type)) +
                     ", query=" +
                     dense_vector_scalar_type_name(runtime.query_vector_metadata.scalar_type));
    }
    if (runtime.base_vector_metadata.cols != runtime.query_vector_metadata.cols) {
        fail_graph("base/query vector dimension mismatch: base dim=" +
                     std::to_string(runtime.base_vector_metadata.cols) +
                     ", query dim=" + std::to_string(runtime.query_vector_metadata.cols));
    }
    runtime.vector_scalar_type = runtime.base_vector_metadata.scalar_type;
    runtime.query_preparation_timing.static_stages_ms["inspect_base_and_validate_vectors"] =
        ns_to_ms(elapsed_load_ns(stage_start)) -
        runtime.query_preparation_timing.query_vector_inspect_ms;
    log_graph_load_stage("inspect_vectors", stage_start);

    stage_start = LoadClock::now();
    const bool custom_rerank_store =
        !go_fann::defaults::rerank_vector_storage_is_default(
            runtime.rerank_vector_storage_mode);
    const bool external_base_payload = custom_rerank_store;
    if (custom_rerank_store) {
        runtime.base_vector_file = inspect_vector_file(config.base_vector_path,
                                                            config.dataset_cutoff);
        validate_vector_file_metadata(runtime.base_vector_file, runtime.base_vector_metadata);
    }
    if (custom_rerank_store && !config.defer_rerank_vector_store) {
        if (!dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
            fail_graph("custom rerank vector stores require float16/float32 base vectors");
        }
        if (go_fann::defaults::rerank_vector_storage_is_mmap_fp16(
                runtime.rerank_vector_storage_mode)) {
            runtime.rerank_vector_store =
                RerankVectorStore::map_fp16(runtime.base_vector_file);
        } else {
            if (runtime.rerank_vector_artifact_path.empty()) {
                fail_graph("memory_int8 requires rerank_vector_artifact_path");
            }
            runtime.rerank_vector_store = RerankVectorStore::load_memory_int8(
                runtime.rerank_vector_artifact_path,
                runtime.base_vector_metadata);
        }
        std::cout << "[rerank-vector-store] mode=" << runtime.rerank_vector_store->mode()
                  << " distance_semantics="
                  << (go_fann::defaults::rerank_vector_storage_is_mmap_fp16(
                          runtime.rerank_vector_storage_mode)
                          ? "fp16_exact"
                          : "int8_approx")
                  << " distance_kernel="
                  << runtime.rerank_vector_store->distance_kernel()
                  << " rows=" << runtime.rerank_vector_store->rows()
                  << " cols=" << runtime.rerank_vector_store->cols()
                  << " mapped_bytes=" << runtime.rerank_vector_store->mapped_bytes()
                  << " resident_payload_bytes="
                  << runtime.rerank_vector_store->resident_payload_bytes()
                  << "\n";
    }
    runtime.query_preparation_timing.static_stages_ms["rerank_vector_store_setup"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("rerank_vector_store_setup", stage_start);

    stage_start = LoadClock::now();
    if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
        if (external_base_payload) {
            runtime.base_vectors.rows = runtime.base_vector_file.rows;
            runtime.base_vectors.cols = runtime.base_vector_file.cols;
        } else {
            runtime.base_vectors = load_uint8_matrix_auto(config.base_vector_path,
                                                          config.dataset_cutoff);
        }
        if (runtime.base_vectors.rows != runtime.base_vector_metadata.rows ||
            runtime.base_vectors.cols != runtime.base_vector_metadata.cols) {
            fail_graph("loaded base vector shape does not match inspected header");
        }
        const auto query_vector_read_start = LoadClock::now();
        runtime.query_vectors = load_uint8_matrix_auto(config.query_vector_path);
        if (runtime.query_vectors.rows != runtime.query_vector_metadata.rows ||
            runtime.query_vectors.cols != runtime.query_vector_metadata.cols) {
            fail_graph("loaded query vector shape does not match inspected header");
        }
        runtime.original_query_vectors = std::move(runtime.query_vectors);
        runtime.query_vectors.rows = runtime.original_query_vectors.rows;
        runtime.query_vectors.cols = runtime.original_query_vectors.cols;
        runtime.query_preparation_timing.query_vector_read_ms =
            ns_to_ms(elapsed_load_ns(query_vector_read_start));
    } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
        if (external_base_payload) {
            runtime.base_float_vectors.rows = runtime.base_vector_file.rows;
            runtime.base_float_vectors.cols = runtime.base_vector_file.cols;
            runtime.base_vectors.rows = runtime.base_vector_file.rows;
            runtime.base_vectors.cols = runtime.base_vector_file.cols;
        } else {
            runtime.base_float_vectors = load_float_matrix_auto(config.base_vector_path,
                                                                config.dataset_cutoff);
        }
        const auto query_vector_read_start = LoadClock::now();
        runtime.query_float_vectors = load_float_matrix_auto(config.query_vector_path);
        if (runtime.base_float_vectors.rows != runtime.base_vector_metadata.rows ||
            runtime.base_float_vectors.cols != runtime.base_vector_metadata.cols ||
            runtime.query_float_vectors.rows != runtime.query_vector_metadata.rows ||
            runtime.query_float_vectors.cols != runtime.query_vector_metadata.cols) {
            fail_graph("loaded float vector shape does not match inspected header");
        }
        runtime.original_query_float_vectors = std::move(runtime.query_float_vectors);
        runtime.query_float_vectors.rows = runtime.original_query_float_vectors.rows;
        runtime.query_float_vectors.cols = runtime.original_query_float_vectors.cols;
        runtime.base_vectors.rows = runtime.base_float_vectors.rows;
        runtime.base_vectors.cols = runtime.base_float_vectors.cols;
        runtime.query_vectors.rows = runtime.query_float_vectors.rows;
        runtime.query_vectors.cols = runtime.query_float_vectors.cols;
        runtime.original_query_vectors.rows = runtime.query_float_vectors.rows;
        runtime.original_query_vectors.cols = runtime.query_float_vectors.cols;
        runtime.query_preparation_timing.query_vector_read_ms =
            ns_to_ms(elapsed_load_ns(query_vector_read_start));
    } else {
        fail_graph("unsupported dense vector scalar type for " + config.base_vector_path);
    }
    require_external_base_payload_not_loaded(runtime, external_base_payload);
    runtime.query_preparation_timing.static_stages_ms["base_vector_read"] =
        ns_to_ms(elapsed_load_ns(stage_start)) -
        runtime.query_preparation_timing.query_vector_read_ms;
    log_graph_load_stage("load_vectors", stage_start);

    stage_start = LoadClock::now();
    uint32_t base_range_rows = 0;
    uint32_t base_range_cols = 0;
    runtime.base_range_values = load_float_fbin_column(config.base_range_fbin_path,
                                                       config.dataset_cutoff,
                                                       &base_range_rows,
                                                       &base_range_cols);
    runtime.base_range_count = base_range_cols == 0 ? 1 : base_range_cols;
    runtime.original_query_count = runtime.query_vectors.rows;
    const SpmatHeaderInfo data_label_header =
        inspect_spmat_header(config.data_label_spmat_path, config.dataset_cutoff);
    runtime.query_preparation_timing.static_stages_ms["range_and_label_header"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("range_and_label_header", stage_start);

    stage_start = LoadClock::now();
    if (config.force_empty_query_labels) {
        if (!config.query_manifest_jsonl_path.empty()) {
            fail_graph("force_empty_query_labels cannot be combined with query_manifest_jsonl_path");
        }
        runtime.query_labels_matrix.rows = runtime.query_vectors.rows;
        runtime.query_labels_matrix.cols = static_cast<uint32_t>(data_label_header.cols);
        runtime.query_labels_matrix.indptr.assign(
            static_cast<size_t>(runtime.query_vectors.rows) + 1, 0);
        runtime.query_labels_matrix.row_labels.resize(runtime.query_vectors.rows);
        runtime.original_query_labels_matrix = std::move(runtime.query_labels_matrix);
        runtime.query_labels_matrix.rows = runtime.original_query_labels_matrix.rows;
        runtime.query_labels_matrix.cols = runtime.original_query_labels_matrix.cols;
        runtime.original_query_label_ops.assign(runtime.original_query_count, QueryLabelOp::And);
        runtime.query_preparation_timing.query_label_materialize_ms =
            ns_to_ms(elapsed_load_ns(stage_start));
    } else if (!config.query_manifest_jsonl_path.empty()) {
        QueryManifestLabels manifest_labels =
            load_query_manifest_jsonl_labels(config,
                                            runtime.query_vectors.rows,
                                            static_cast<uint32_t>(data_label_header.cols),
                                            &runtime.query_preparation_timing);
        const auto label_materialize_start = LoadClock::now();
        runtime.query_labels_matrix = std::move(manifest_labels.labels);
        runtime.original_query_labels_matrix = std::move(runtime.query_labels_matrix);
        runtime.query_labels_matrix.rows = runtime.original_query_labels_matrix.rows;
        runtime.query_labels_matrix.cols = runtime.original_query_labels_matrix.cols;
        runtime.original_query_label_ops = std::move(manifest_labels.ops);
        runtime.original_query_clauses = std::move(manifest_labels.clauses);
        runtime.query_preparation_timing.query_label_materialize_ms =
            ns_to_ms(elapsed_load_ns(label_materialize_start));
    } else {
        const auto query_label_read_start = LoadClock::now();
        runtime.query_labels_matrix = load_spmat_labels(config.query_label_spmat_path,
                                                        runtime.query_vectors.rows,
                                                        true);
        runtime.original_query_labels_matrix = std::move(runtime.query_labels_matrix);
        runtime.query_labels_matrix.rows = runtime.original_query_labels_matrix.rows;
        runtime.query_labels_matrix.cols = runtime.original_query_labels_matrix.cols;
        runtime.original_query_label_ops.assign(runtime.original_query_count, QueryLabelOp::And);
        runtime.query_preparation_timing.query_label_spmat_read_ms =
            ns_to_ms(elapsed_load_ns(query_label_read_start));
    }
    log_graph_load_stage(config.query_manifest_jsonl_path.empty()
                               ? "load_query_label_spmat"
                               : "load_query_manifest",
                           stage_start);

    stage_start = LoadClock::now();
    const auto query_range_read_start = LoadClock::now();
    runtime.original_query_ranges = load_query_range_binary(config.query_range_path, runtime.query_vectors.rows);
    runtime.query_preparation_timing.query_range_read_ms =
        ns_to_ms(elapsed_load_ns(query_range_read_start));
    if (runtime.base_vectors.rows != data_label_header.effective_rows) {
        fail_graph("base vectors and data labels row count mismatch");
    }
    runtime.group_layout =
        load_group_layout(config.group_path, static_cast<uint32_t>(data_label_header.cols));
    if (config.implicit_all_ones_row_filter_mask) {
        if (runtime.group_layout.groups.size() != data_label_header.cols ||
            runtime.group_layout.label_to_group.size() != data_label_header.cols) {
            fail_graph("implicit all-ones row filter mask requires identity group dimensions");
        }
        for (uint32_t group = 0; group < runtime.group_layout.groups.size(); ++group) {
            const auto& labels = runtime.group_layout.groups[group];
            if (labels.size() != 1 || labels.front() != group ||
                runtime.group_layout.label_to_group[group] != group) {
                fail_graph("implicit all-ones row filter mask requires identity group layout");
            }
        }
    }
    runtime.label_intervals =
        load_range_label_manifest(config.range_label_manifest_path, static_cast<uint32_t>(data_label_header.cols));
    runtime.query_preparation_timing.static_stages_ms["group_layout_and_range_manifest"] =
        ns_to_ms(elapsed_load_ns(stage_start)) -
        runtime.query_preparation_timing.query_range_read_ms;
    log_graph_load_stage("load_query_range_and_group_layout", stage_start);

    if (!runtime.base_range_values.empty() &&
        runtime.base_range_values.size() != static_cast<size_t>(runtime.base_vectors.rows) * runtime.base_range_count) {
        fail_graph("base range values row count mismatch");
    }
    if (!runtime.base_range_values.empty() && base_range_rows != runtime.base_vectors.rows) {
        fail_graph("base range values row count mismatch");
    }
    if (!runtime.base_range_values.empty() &&
        runtime.original_query_ranges.range_count != runtime.base_range_count) {
        fail_graph("query/base range dimension mismatch");
    }
    if (!config.range_label_manifest_path.empty() &&
        !runtime.base_range_values.empty() &&
        runtime.label_intervals.range_count != runtime.base_range_count) {
        fail_graph("range label manifest/base range dimension mismatch");
    }
    for (uint32_t query_idx = 0; query_idx < runtime.original_query_ranges.rows; ++query_idx) {
        if (query_has_any_active_range(runtime.original_query_ranges, query_idx) && runtime.base_range_values.empty()) {
            fail_graph("active range queries require base_range_fbin_path");
        }
    }
    if (runtime.base_vector_metadata.cols != static_cast<uint32_t>(config.dim)) {
        fail_graph("base vector dimension mismatch: file dim=" +
                     std::to_string(runtime.base_vector_metadata.cols) +
                     ", configured dim=" + std::to_string(config.dim));
    }
    if (runtime.query_vector_metadata.cols != static_cast<uint32_t>(config.dim)) {
        fail_graph("query vector dimension mismatch: file dim=" +
                     std::to_string(runtime.query_vector_metadata.cols) +
                     ", configured dim=" + std::to_string(config.dim));
    }
    if (!config.allow_float_vectors) {
        require_uint8_graph_runtime(runtime, config.vector_consumer);
    }

    stage_start = LoadClock::now();
    const std::string graph_file_path = std::filesystem::exists(config.graph_path)
        ? config.graph_path : default_graph_file_path_from_graph_path(config.graph_path);
    runtime.graph = load_graph_matrix_auto(graph_file_path);
    runtime.query_preparation_timing.static_stages_ms["load_graph"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("load_graph", stage_start);

    stage_start = LoadClock::now();
    const std::string lowfreq_index_path = config.lowfreq_index_path.empty()
        ? default_lowfreq_index_path_from_graph_path(config.graph_path)
        : config.lowfreq_index_path;
    if (std::filesystem::exists(lowfreq_index_path)) {
        runtime.lowfreq_index = load_lowfreq_posting_index(lowfreq_index_path);
    }
    runtime.query_preparation_timing.static_stages_ms["load_lowfreq_index"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("load_lowfreq_index", stage_start);

    stage_start = LoadClock::now();
    const uint64_t effective_row_limit =
        config.dataset_cutoff > 0 ? config.dataset_cutoff : data_label_header.rows;
    const bool metadata_loaded =
        load_metadata_sidecar(config.metadata_sidecar_path,
                              runtime.base_vectors.rows,
                              effective_row_limit,
                              runtime.graph.rows,
                              config.specificity_threshold,
                              data_label_header,
                              runtime.group_layout,
                              runtime.metadata,
                              config.implicit_all_ones_row_filter_mask);
    runtime.query_preparation_timing.static_stages_ms["load_metadata_sidecar"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("load_metadata_sidecar", stage_start);

    stage_start = LoadClock::now();
    if (!metadata_loaded || config.load_data_labels_for_filtering) {
        runtime.data_labels = load_spmat_labels(config.data_label_spmat_path, config.dataset_cutoff);
        if (runtime.base_vectors.rows != runtime.data_labels.rows) {
            fail_graph("base vectors and data labels row count mismatch");
        }
    } else {
        runtime.data_labels.rows = runtime.base_vectors.rows;
        runtime.data_labels.cols = static_cast<uint32_t>(data_label_header.cols);
    }
    runtime.query_preparation_timing.static_stages_ms["load_data_labels"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("load_data_labels", stage_start);

    stage_start = LoadClock::now();
    if (!metadata_loaded) {
        runtime.metadata = rebuild_graph_metadata(runtime.data_labels,
                                                   config.specificity_threshold,
                                                   runtime.group_layout);
        save_metadata_sidecar(config.metadata_sidecar_path,
                              runtime.metadata,
                              runtime.base_vectors.rows,
                              effective_row_limit,
                              runtime.graph.rows,
                              runtime.graph.cols,
                              config.specificity_threshold,
                              runtime.group_layout);
    }
    runtime.metadata.implicit_all_ones_row_filter_mask =
        config.implicit_all_ones_row_filter_mask;
    if (runtime.metadata.implicit_all_ones_row_filter_mask) {
        runtime.metadata.row_filter_mask.clear();
        runtime.metadata.row_filter_mask.shrink_to_fit();
    }
    runtime.query_preparation_timing.static_stages_ms["rebuild_metadata"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("rebuild_metadata", stage_start);
    if (runtime.base_vectors.cols != static_cast<uint32_t>(config.dim)) {
        fail_graph("base vector dimension mismatch: file dim=" +
                     std::to_string(runtime.base_vectors.cols) +
                     ", configured dim=" + std::to_string(config.dim));
    }
    if (runtime.graph.rows != runtime.metadata.total_rows) {
        fail_graph("ivf_graph.bin rows do not match reconstructed high-frequency rows");
    }
    if (runtime.graph.cols > static_cast<uint64_t>(config.degree)) {
        fail_graph("graph degree exceeds configured search degree");
    }

    stage_start = LoadClock::now();
    if (!load_seed_sidecar(config.seed_sidecar_path,
                           runtime.metadata,
                           runtime.base_vectors.rows,
                           runtime.graph.cols,
                           config.specificity_threshold,
                           runtime.seed_local_id_by_label)) {
        if (external_base_payload) {
            fail_graph("external rerank vectors require a compatible seed_sidecar_path; "
                         "runtime seed fallback needs full base vectors and is disabled in external-vector mode. "
                         "Generate matching seed sidecars and pass --seed-sidecar-path explicitly if config.json is stale");
        }
        if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
            runtime.seed_local_id_by_label =
                build_seed_local_ids(runtime.base_float_vectors, runtime.metadata, config.dim);
        } else {
            runtime.seed_local_id_by_label =
                build_seed_local_ids(runtime.base_vectors, runtime.metadata, config.dim);
        }
        save_seed_sidecar(config.seed_sidecar_path,
                          runtime.metadata,
                          runtime.base_vectors.rows,
                          runtime.graph.cols,
                          config.specificity_threshold,
                          runtime.seed_local_id_by_label);
    }
    runtime.query_preparation_timing.static_stages_ms["load_seed_sidecar"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("load_seed_sidecar", stage_start);

    stage_start = LoadClock::now();
    if (!load_original_label_seed_sidecar(config.original_label_seed_sidecar_path,
                                          runtime.metadata,
                                          runtime.group_layout,
                                          runtime.base_vectors.rows,
                                          runtime.graph.cols,
                                          config.specificity_threshold,
                                          runtime.seed_by_original_label)) {
        if (external_base_payload) {
            fail_graph("external rerank vectors require a compatible original_label_seed_sidecar_path; "
                         "runtime seed fallback needs full base vectors and is disabled in external-vector mode. "
                         "Generate matching original-label seed sidecars and pass "
                         "--original-label-seed-sidecar-path explicitly if config.json is stale");
        }
        if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
            runtime.seed_by_original_label =
                build_seeds_by_original_label(runtime.base_float_vectors,
                                              runtime.metadata,
                                              runtime.group_layout,
                                              config.dim);
        } else {
            runtime.seed_by_original_label =
                build_seeds_by_original_label(runtime.base_vectors,
                                              runtime.metadata,
                                              runtime.group_layout,
                                              config.dim);
        }
        save_original_label_seed_sidecar(config.original_label_seed_sidecar_path,
                                         runtime.metadata,
                                         runtime.group_layout,
                                         runtime.base_vectors.rows,
                                         runtime.graph.cols,
                                         config.specificity_threshold,
                                         runtime.seed_by_original_label);
    }
    runtime.query_preparation_timing.static_stages_ms["load_original_label_seed_sidecar"] =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("load_original_label_seed_sidecar", stage_start);

    runtime.seed_selection_mode = config.seed_selection_mode;

    stage_start = LoadClock::now();
    const double previous_cardinality_ms =
        runtime.query_preparation_timing.static_label_cardinality_init_ms;
    validate_and_prepare_queries(runtime,
                                 config.dim,
                                 config.routing_mode,
                                 config.original_query_route_mask,
                                 config.supports_shared_label_routes);
    runtime.query_preparation_timing.route_generation_ms =
        ns_to_ms(elapsed_load_ns(stage_start)) -
        (runtime.query_preparation_timing.static_label_cardinality_init_ms - previous_cardinality_ms);
    log_graph_load_stage("route_queries", stage_start);
    if (!config.range_posting_index_path.empty()) {
        stage_start = LoadClock::now();
        runtime.range_posting_index =
            std::make_shared<go_fann::range_posting::Index>(
                go_fann::range_posting::Index::load(
                    config.range_posting_index_path));
        runtime.query_preparation_timing.static_range_posting_index_load_ms =
            ns_to_ms(elapsed_load_ns(stage_start));
        stage_start = LoadClock::now();
        plan_graph_range_access(runtime,
                                  *runtime.range_posting_index,
                                  config.range_access_planner);
        runtime.query_preparation_timing.range_access_plan_ms =
            ns_to_ms(elapsed_load_ns(stage_start));
        log_graph_load_stage("plan_range_access", stage_start);
    }
    stage_start = LoadClock::now();
    runtime.query_preparation_timing.route_select_emit_ms = runtime.query_preparation_timing.route_generation_ms;
    if (config.supports_shared_label_routes) {
        const auto union_start = LoadClock::now();
        coalesce_graph_routes(runtime);
        runtime.query_preparation_timing.route_union_ms = ns_to_ms(elapsed_load_ns(union_start));
        const auto payload_start = LoadClock::now();
        materialize_graph_route_payloads(runtime);
        runtime.query_preparation_timing.route_payload_materialize_ms = ns_to_ms(elapsed_load_ns(payload_start));
    }
    runtime.query_preparation_timing.route_generation_ms += ns_to_ms(elapsed_load_ns(stage_start));
    stage_start = LoadClock::now();
    ensure_host_candidate_label_data(runtime, config);
    runtime.query_preparation_timing.host_candidate_label_load_ms =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("load_host_candidate_labels", stage_start);
#ifndef NDEBUG
    validate_graph_runtime_query_metadata(runtime, "load_graph_runtime(after validate_and_prepare_queries)");
#endif
    trace_graph_runtime_summary(runtime, "after validate_and_prepare_queries");
    if (dense_vector_scalar_is_float(runtime.vector_scalar_type) &&
        !runtime.high_query_indices.empty() &&
        config.highfreq_prefetch_with_vector_payload) {
        fail_graph(config.vector_consumer +
                     " cannot publish high-frequency vector payloads for float dense vectors; "
                     "use a CPU/off high-frequency rerank mode or route the queries through the low-frequency path");
    }
    stage_start = LoadClock::now();
    if (!runtime.high_query_indices.empty() && config.routing_mode != "greedy_max_posting") {
        AsyncIoConfig io_config(static_cast<uint32_t>(config.dim),
                                static_cast<uint32_t>(config.degree),
                                static_cast<uint32_t>(config.prefetch_hops));
        io_config.publish_valid_vector_payload = config.highfreq_prefetch_with_vector_payload;
        io_config.skip_filtered_vector_payload = !config.highfreq_prefetch_with_vector_payload;
        io_config.include_sample_ids = config.highfreq_prefetch_with_sample_ids;
        io_config.set_reserve_vector_payload(config.highfreq_prefetch_with_vector_payload);
        build_prefetch_slices(runtime, io_config);
    } else {
        runtime.prefetch_nodes.clear();
        runtime.prefetch_index.clear();
        runtime.prefetch_begin_by_group.assign(runtime.metadata.label_count, 0);
        runtime.prefetch_end_by_group.assign(runtime.metadata.label_count, 0);
    }
    runtime.query_preparation_timing.prefetch_build_ms =
        ns_to_ms(elapsed_load_ns(stage_start));
    log_graph_load_stage("build_prefetch", stage_start);
    runtime.query_preparation_timing.runtime_load_total_ms =
        ns_to_ms(elapsed_load_ns(total_start));
    log_graph_load_stage("total", total_start);
#ifndef NDEBUG
    validate_graph_runtime_query_metadata(runtime, "load_graph_runtime(final)");
#endif
    return runtime;
}

void swap_graph_query_state(GraphRuntime& runtime,
                              GraphQueryState& query_state) noexcept
{
    using std::swap;
    swap(runtime.route_payload_preparation, query_state.route_payload_preparation);
    swap(runtime.shared_route_query_vectors, query_state.shared_route_query_vectors);
    swap(runtime.query_vector_metadata, query_state.query_vector_metadata);
    swap(runtime.original_query_vectors, query_state.original_query_vectors);
    swap(runtime.query_vectors, query_state.query_vectors);
    swap(runtime.original_query_float_vectors, query_state.original_query_float_vectors);
    swap(runtime.query_float_vectors, query_state.query_float_vectors);
    swap(runtime.original_query_count, query_state.original_query_count);
    swap(runtime.original_query_labels_matrix, query_state.original_query_labels_matrix);
    swap(runtime.original_query_label_ops, query_state.original_query_label_ops);
    swap(runtime.original_query_clauses, query_state.original_query_clauses);
    swap(runtime.query_labels_matrix, query_state.query_labels_matrix);
    swap(runtime.original_query_ranges, query_state.original_query_ranges);
    swap(runtime.query_ranges, query_state.query_ranges);
    swap(runtime.query_group_ids, query_state.query_group_ids);
    swap(runtime.query_filter_labels, query_state.query_filter_labels);
    swap(runtime.query_filter_bits, query_state.query_filter_bits);
    swap(runtime.start_local_ids, query_state.start_local_ids);
    swap(runtime.query_is_lowfreq, query_state.query_is_lowfreq);
    swap(runtime.route_to_original_query, query_state.route_to_original_query);
    swap(runtime.route_to_clause, query_state.route_to_clause);
    swap(runtime.route_clause_indptr, query_state.route_clause_indptr);
    swap(runtime.route_clause_ids, query_state.route_clause_ids);
    swap(runtime.route_top_branch_ids, query_state.route_top_branch_ids);
    swap(runtime.route_enabled, query_state.route_enabled);
    swap(runtime.route_providers, query_state.route_providers);
    swap(runtime.route_execution_modes, query_state.route_execution_modes);
    swap(runtime.route_candidate_sources, query_state.route_candidate_sources);
    swap(runtime.route_range_posting_begins, query_state.route_range_posting_begins);
    swap(runtime.route_range_posting_sizes, query_state.route_range_posting_sizes);
    swap(runtime.route_range_posting_indices, query_state.route_range_posting_indices);
    swap(runtime.range_graph_and_original_queries,
         query_state.range_graph_and_original_queries);
    swap(runtime.range_graph_or_original_queries,
         query_state.range_graph_or_original_queries);
    swap(runtime.range_graph_clause_indptr,
         query_state.range_graph_clause_indptr);
    swap(runtime.range_graph_clause_ids,
         query_state.range_graph_clause_ids);
    swap(runtime.high_query_indices, query_state.high_query_indices);
    swap(runtime.low_query_indices, query_state.low_query_indices);
    swap(runtime.or_route_planner_profile, query_state.or_route_planner_profile);
    swap(runtime.range_access_planner_profile, query_state.range_access_planner_profile);
    swap(runtime.query_preparation_timing, query_state.query_preparation_timing);
}

static GraphQueryState load_graph_query_state_impl(GraphRuntime& runtime,
                                                       const GraphLoadConfig& config,
                                                       bool planning_only)
{
    const auto total_start = LoadClock::now();
    GraphQueryState original_state;
    swap_graph_query_state(runtime, original_state);
    runtime.shared_route_query_vectors = config.supports_shared_query_vectors && snapshot_query_preparation_config().optimized;

    try {
        auto stage_start = LoadClock::now();
        runtime.query_vector_metadata = inspect_dense_vector_file(config.query_vector_path);
        if (runtime.base_vector_metadata.scalar_type != runtime.query_vector_metadata.scalar_type) {
            fail_graph("base/warmup query vector scalar type mismatch: base=" +
                         std::string(dense_vector_scalar_type_name(runtime.base_vector_metadata.scalar_type)) +
                         ", warmup query=" +
                         dense_vector_scalar_type_name(runtime.query_vector_metadata.scalar_type));
        }
        if (runtime.base_vector_metadata.cols != runtime.query_vector_metadata.cols) {
            fail_graph("base/warmup query vector dimension mismatch: base dim=" +
                         std::to_string(runtime.base_vector_metadata.cols) +
                         ", warmup query dim=" +
                         std::to_string(runtime.query_vector_metadata.cols));
        }

        runtime.query_preparation_timing.query_vector_inspect_ms =
            ns_to_ms(elapsed_load_ns(stage_start));
        stage_start = LoadClock::now();
        if (runtime.vector_scalar_type == DenseVectorScalarType::Uint8) {
            runtime.query_vectors = load_uint8_matrix_auto(config.query_vector_path);
            runtime.original_query_vectors = std::move(runtime.query_vectors);
        runtime.query_vectors.rows = runtime.original_query_vectors.rows;
        runtime.query_vectors.cols = runtime.original_query_vectors.cols;
        } else if (dense_vector_scalar_is_float(runtime.vector_scalar_type)) {
            runtime.query_float_vectors = load_float_matrix_auto(config.query_vector_path);
            runtime.original_query_float_vectors = std::move(runtime.query_float_vectors);
        runtime.query_float_vectors.rows = runtime.original_query_float_vectors.rows;
        runtime.query_float_vectors.cols = runtime.original_query_float_vectors.cols;
            runtime.query_vectors.rows = runtime.query_float_vectors.rows;
            runtime.query_vectors.cols = runtime.query_float_vectors.cols;
            runtime.original_query_vectors.rows = runtime.query_float_vectors.rows;
            runtime.original_query_vectors.cols = runtime.query_float_vectors.cols;
        } else {
            fail_graph("unsupported warmup query vector scalar type for " +
                         config.query_vector_path);
        }
        if (runtime.query_vectors.rows != runtime.query_vector_metadata.rows ||
            runtime.query_vectors.cols != runtime.query_vector_metadata.cols) {
            fail_graph("loaded warmup query vector shape does not match inspected header");
        }

        runtime.query_preparation_timing.query_vector_read_ms =
            ns_to_ms(elapsed_load_ns(stage_start));
        runtime.original_query_count = runtime.query_vectors.rows;
        const uint32_t original_label_count = runtime.data_labels.cols;
        if (original_label_count == 0) {
            fail_graph("warmup query loading requires initialized data-label metadata");
        }
        if (config.force_empty_query_labels) {
            if (!config.query_manifest_jsonl_path.empty())
                fail_graph("empty query labels cannot be combined with a manifest");
            runtime.original_query_labels_matrix.rows = runtime.original_query_count;
            runtime.original_query_labels_matrix.cols = original_label_count;
            runtime.original_query_labels_matrix.indptr.assign(runtime.original_query_count + 1, 0);
            runtime.query_labels_matrix.cols = original_label_count;
            runtime.original_query_label_ops.assign(runtime.original_query_count, QueryLabelOp::And);
        } else if (!config.query_manifest_jsonl_path.empty()) {
            QueryManifestLabels manifest_labels =
                load_query_manifest_jsonl_labels(config,
                                                 runtime.query_vectors.rows,
                                                 original_label_count,
                                                 &runtime.query_preparation_timing);
            stage_start = LoadClock::now();
            runtime.query_labels_matrix = std::move(manifest_labels.labels);
            runtime.original_query_labels_matrix = std::move(runtime.query_labels_matrix);
        runtime.query_labels_matrix.rows = runtime.original_query_labels_matrix.rows;
        runtime.query_labels_matrix.cols = runtime.original_query_labels_matrix.cols;
            runtime.original_query_label_ops = std::move(manifest_labels.ops);
            runtime.original_query_clauses = std::move(manifest_labels.clauses);
            runtime.query_preparation_timing.query_label_materialize_ms =
                ns_to_ms(elapsed_load_ns(stage_start));
        } else {
            stage_start = LoadClock::now();
            runtime.query_labels_matrix = load_spmat_labels(config.query_label_spmat_path,
                                                            runtime.query_vectors.rows,
                                                            true);
            runtime.original_query_labels_matrix = std::move(runtime.query_labels_matrix);
        runtime.query_labels_matrix.rows = runtime.original_query_labels_matrix.rows;
        runtime.query_labels_matrix.cols = runtime.original_query_labels_matrix.cols;
            runtime.original_query_label_ops.assign(runtime.original_query_count,
                                                    QueryLabelOp::And);
            runtime.query_preparation_timing.query_label_spmat_read_ms =
                ns_to_ms(elapsed_load_ns(stage_start));
        }
        if (runtime.query_labels_matrix.cols != original_label_count) {
            fail_graph("warmup query labels column count does not match data labels");
        }

        stage_start = LoadClock::now();
        runtime.original_query_ranges = load_query_range_binary(config.query_range_path,
                                                                runtime.query_vectors.rows);
        runtime.query_preparation_timing.query_range_read_ms =
            ns_to_ms(elapsed_load_ns(stage_start));
        if (runtime.original_query_ranges.range_count != runtime.base_range_count) {
            fail_graph("warmup query/base range dimension mismatch");
        }
        for (uint32_t query_idx = 0; query_idx < runtime.original_query_ranges.rows; ++query_idx) {
            if (query_has_any_active_range(runtime.original_query_ranges, query_idx) &&
                runtime.base_range_values.empty() && !planning_only) {
                fail_graph("active warmup range queries require base_range_fbin_path");
            }
        }
        if (runtime.query_vector_metadata.cols != static_cast<uint32_t>(config.dim)) {
            fail_graph("warmup query vector dimension mismatch: file dim=" +
                         std::to_string(runtime.query_vector_metadata.cols) +
                         ", configured dim=" + std::to_string(config.dim));
        }

        stage_start = LoadClock::now();
        const double previous_cardinality_ms =
            runtime.query_preparation_timing.static_label_cardinality_init_ms;
        validate_and_prepare_queries(runtime,
                                     config.dim,
                                     config.routing_mode,
                                     config.original_query_route_mask,
                                     config.supports_shared_label_routes);
        runtime.query_preparation_timing.route_generation_ms =
            ns_to_ms(elapsed_load_ns(stage_start)) -
            (runtime.query_preparation_timing.static_label_cardinality_init_ms - previous_cardinality_ms);
        if (!config.range_posting_index_path.empty()) {
            if (runtime.range_posting_index == nullptr) {
                fail_graph("warmup range access planning requires the primary range index");
            }
            stage_start = LoadClock::now();
            plan_graph_range_access(runtime,
                                      *runtime.range_posting_index,
                                      config.range_access_planner);
            runtime.query_preparation_timing.range_access_plan_ms =
                ns_to_ms(elapsed_load_ns(stage_start));
        }
        stage_start = LoadClock::now();
        runtime.query_preparation_timing.route_select_emit_ms = runtime.query_preparation_timing.route_generation_ms;
        if (config.supports_shared_label_routes) {
            const auto union_start = LoadClock::now();
            coalesce_graph_routes(runtime);
            runtime.query_preparation_timing.route_union_ms = ns_to_ms(elapsed_load_ns(union_start));
            const auto payload_start = LoadClock::now();
            materialize_graph_route_payloads(runtime);
            runtime.query_preparation_timing.route_payload_materialize_ms = ns_to_ms(elapsed_load_ns(payload_start));
        }
        runtime.query_preparation_timing.route_generation_ms += ns_to_ms(elapsed_load_ns(stage_start));
        stage_start = LoadClock::now();
        if (!planning_only) ensure_host_candidate_label_data(runtime, config);
        runtime.query_preparation_timing.host_candidate_label_load_ms =
            ns_to_ms(elapsed_load_ns(stage_start));
#ifndef NDEBUG
        validate_graph_runtime_query_metadata(runtime, "load_graph_query_state");
#endif

        runtime.query_preparation_timing.runtime_load_total_ms =
            ns_to_ms(elapsed_load_ns(total_start));
        GraphQueryState loaded_state;
        swap_graph_query_state(runtime, loaded_state);
        swap_graph_query_state(runtime, original_state);
        return loaded_state;
    } catch (...) {
        swap_graph_query_state(runtime, original_state);
        throw;
    }
}


GraphQueryState load_graph_query_state(GraphRuntime& runtime,
                                          const GraphLoadConfig& config)
{
    return load_graph_query_state_impl(runtime, config, false);
}

GraphQueryState prepare_graph_query_batch(GraphRoutingCatalog& catalog,
                                             const GraphLoadConfig& config)
{
    return load_graph_query_state_impl(catalog.metadata_, config, true);
}

GraphRoutingCatalog load_graph_routing_catalog(
    const GraphLoadConfig& config, const std::vector<uint32_t>& exact_label_cardinalities)
{
    GraphRoutingCatalog catalog;
    auto& runtime = catalog.metadata_;
    runtime.base_vector_metadata = inspect_dense_vector_file(config.base_vector_path, config.dataset_cutoff);
    runtime.vector_scalar_type = runtime.base_vector_metadata.scalar_type;
    runtime.base_vectors.rows = runtime.base_vector_metadata.rows;
    runtime.base_vectors.cols = runtime.base_vector_metadata.cols;
    runtime.supported_query_route_providers = config.supported_query_route_providers;
    runtime.and_dual_route_posting_ratio_threshold = config.and_dual_route_posting_ratio_threshold;
    runtime.seed_selection_mode = config.seed_selection_mode;
    const auto label_header = inspect_spmat_header(config.data_label_spmat_path, config.dataset_cutoff);
    runtime.data_labels.rows = label_header.effective_rows;
    runtime.data_labels.cols = label_header.cols;
    runtime.group_layout = load_group_layout(config.group_path, runtime.data_labels.cols);
    runtime.label_intervals = load_range_label_manifest(config.range_label_manifest_path, runtime.data_labels.cols);
    if (runtime.base_vectors.rows != runtime.data_labels.rows || runtime.base_vectors.cols != static_cast<uint32_t>(config.dim))
        fail_graph("routing catalog base/label shape mismatch");
    std::ifstream input(config.metadata_sidecar_path, std::ios::binary);
    GraphMetadataSidecarHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    const uint64_t row_limit = config.dataset_cutoff ? config.dataset_cutoff : label_header.rows;
    if (!input || header.version != kGraphMetadataSidecarVersion ||
        header.original_label_count != label_header.cols || header.num_points != label_header.effective_rows ||
        header.label_count != runtime.group_layout.groups.size() || header.group_count != header.label_count ||
        header.row_limit != row_limit || header.total_rows != header.graph_rows ||
        header.total_rows > UINT32_MAX || header.group_layout_checksum != group_layout_checksum(runtime.group_layout) ||
        header.specificity_threshold != static_cast<uint64_t>(config.specificity_threshold))
        fail_graph("routing catalog requires a compatible metadata sidecar; no row-scan fallback");
    auto& metadata = runtime.metadata;
    metadata.label_count = header.label_count;
    metadata.original_label_count = header.original_label_count;
    metadata.total_rows = header.total_rows;
    metadata.specificity_threshold = header.specificity_threshold;
    // Graph degree comes from its header, not the metadata sidecar's legacy field.
    const std::string graph_path = std::filesystem::exists(config.graph_path)
        ? config.graph_path : default_graph_file_path_from_graph_path(config.graph_path);
    if (go_fann::graph_file::has_graph_file_magic(graph_path)) {
        const auto graph = go_fann::graph_file::read_graph_file_header(graph_path);
        runtime.graph.rows = graph.rows;
        runtime.graph.cols = graph.degree;
    } else {
        std::ifstream graph_input(graph_path, std::ios::binary);
        graph_input.read(reinterpret_cast<char*>(&runtime.graph.rows), sizeof(runtime.graph.rows));
        graph_input.read(reinterpret_cast<char*>(&runtime.graph.cols), sizeof(runtime.graph.cols));
        if (!graph_input) fail_graph("routing catalog invalid graph matrix header");
    }
    if (runtime.graph.rows != header.graph_rows || runtime.graph.cols == 0 ||
        runtime.graph.cols > static_cast<uint32_t>(config.degree))
        fail_graph("routing catalog graph generation mismatch");
    if (!read_vector_payload(input, config.metadata_sidecar_path, "cat_freq", metadata.label_count, metadata.cat_freq) ||
        !read_vector_payload(input, config.metadata_sidecar_path, "group_posting_size", metadata.label_count, metadata.group_posting_size) ||
        !read_vector_payload(input, config.metadata_sidecar_path, "label_size", metadata.label_count, metadata.label_size) ||
        !read_vector_payload(input, config.metadata_sidecar_path, "label_offset", metadata.label_count, metadata.label_offset))
        fail_graph("routing catalog truncated group metadata");
    const uint64_t mask_offset = static_cast<uint64_t>(input.tellg()) + metadata.total_rows * sizeof(uint32_t);
    const uint64_t bits_offset = mask_offset + metadata.total_rows * sizeof(uint64_t);
    if (bits_offset + metadata.original_label_count * sizeof(uint64_t) != std::filesystem::file_size(config.metadata_sidecar_path))
        fail_graph("routing catalog metadata payload size mismatch");
    input.seekg(bits_offset);
    if (!read_vector_payload(input, config.metadata_sidecar_path, "label_filter_bit", metadata.original_label_count, metadata.label_filter_bit))
        fail_graph("routing catalog missing label filter bits");
    uint64_t graph_rows = 0;
    for (uint32_t group = 0; group < metadata.label_count; ++group) {
        if (metadata.label_offset[group] != graph_rows || metadata.cat_freq[group] != metadata.group_posting_size[group] ||
            (metadata.label_size[group] && (metadata.label_size[group] != metadata.group_posting_size[group] ||
                metadata.label_size[group] <= static_cast<uint32_t>(config.specificity_threshold))))
            fail_graph("routing catalog invalid group sizes/offsets");
        graph_rows += metadata.label_size[group];
    }
    if (graph_rows != metadata.total_rows) fail_graph("routing catalog graph row sum mismatch");
    metadata.original_label_cardinality = exact_label_cardinalities;
    if (metadata.original_label_cardinality.empty()) {
        metadata.original_label_cardinality.assign(metadata.original_label_count, 0);
        for (uint32_t group = 0; group < metadata.label_count; ++group) {
            if (runtime.group_layout.groups[group].size() != 1)
                fail_graph("grouped routing catalog requires explicit exact original-label cardinalities");
            metadata.original_label_cardinality[runtime.group_layout.groups[group][0]] = metadata.group_posting_size[group];
        }
    }
    ensure_exact_original_label_cardinality(runtime);
    const auto lowfreq_path = config.lowfreq_index_path.empty()
        ? default_lowfreq_index_path_from_graph_path(config.graph_path) : config.lowfreq_index_path;
    std::ifstream lowfreq(lowfreq_path, std::ios::binary);
    if (lowfreq) {
        auto& index = runtime.lowfreq_index;
        lowfreq.read(reinterpret_cast<char*>(&index.group_count), sizeof(index.group_count));
        lowfreq.read(reinterpret_cast<char*>(&index.total_rows), sizeof(index.total_rows));
        if (!lowfreq || index.group_count != metadata.label_count ||
            !read_vector_payload(lowfreq, lowfreq_path, "label_size", index.group_count, index.label_size) ||
            !read_vector_payload(lowfreq, lowfreq_path, "label_offset", index.group_count, index.label_offset) ||
            std::filesystem::file_size(lowfreq_path) != 8ULL + 8ULL * index.group_count + 12ULL * index.total_rows)
            fail_graph("routing catalog low-frequency metadata mismatch");
    }
    if (!load_seed_sidecar(config.seed_sidecar_path, metadata, runtime.base_vectors.rows,
                           runtime.graph.cols, config.specificity_threshold, runtime.seed_local_id_by_label))
        fail_graph("routing catalog requires compatible group seed sidecar");
    std::ifstream seeds(config.original_label_seed_sidecar_path, std::ios::binary);
    GraphOriginalLabelSeedHeader seed_header{};
    seeds.read(reinterpret_cast<char*>(&seed_header), sizeof(seed_header));
    if (!seeds || seed_header.version != 2 || seed_header.original_label_count != metadata.original_label_count ||
        seed_header.group_count != metadata.label_count || seed_header.num_points != runtime.base_vectors.rows ||
        seed_header.graph_rows != runtime.graph.rows || seed_header.graph_degree != runtime.graph.cols ||
        seed_header.specificity_threshold != static_cast<uint64_t>(config.specificity_threshold) ||
        !read_vector_payload(seeds, config.original_label_seed_sidecar_path, "seeds", metadata.original_label_count, runtime.seed_by_original_label))
        fail_graph("routing catalog requires compatible original-label seeds");
    for (uint32_t label = 0; label < metadata.original_label_count; ++label) {
        const uint32_t group = runtime.group_layout.label_to_group[label];
        const uint32_t seed = runtime.seed_by_original_label[label];
        if (!metadata.label_size[group] || !metadata.original_label_cardinality[label]) {
            if (seed != UINT32_MAX) fail_graph("routing catalog unexpected seed for empty label");
            continue;
        }
        if (seed >= metadata.label_size[group]) fail_graph("routing catalog seed outside group");
        // Validate only the selected seed's predicate bit, without loading the
        // full graph-row mask payload. Identity groups imply their sole bit.
        if (runtime.group_layout.groups[group].size() > 1) {
            uint64_t mask = 0;
            input.seekg(mask_offset + (static_cast<uint64_t>(metadata.label_offset[group]) + seed) * sizeof(mask));
            input.read(reinterpret_cast<char*>(&mask), sizeof(mask));
            if (!input || !(mask & metadata.label_filter_bit[label])) fail_graph("routing catalog seed predicate mismatch");
        }
    }
    if (!config.base_range_fbin_path.empty()) {
        const auto ranges = inspect_dense_vector_file(config.base_range_fbin_path, config.dataset_cutoff);
        if (ranges.rows != runtime.base_vectors.rows) fail_graph("routing catalog range/base count mismatch");
        runtime.base_range_count = ranges.cols;
    }
    if (!config.range_posting_index_path.empty())
        runtime.range_posting_index = std::make_shared<go_fann::range_posting::Index>(
            go_fann::range_posting::Index::load(config.range_posting_index_path));
    return catalog;
}
