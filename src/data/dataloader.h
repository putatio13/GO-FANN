#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "data/node_layout.h"
#include "index/graph_file_format.h"
#include "app/runtime_defaults.h"

namespace go_fann::range_posting {
class Index;
}

using value_t = u_int8_t;

constexpr uint32_t kGraphSeedVersion = 2;
constexpr uint32_t kGraphMetadataSidecarVersion = 1;

enum class DenseVectorScalarType : uint8_t {
    Unknown = 0,
    Uint8 = 1,
    Float32 = 2,
    Float16 = 3,
};

enum class QueryLabelOp : uint8_t {
    And = 0,
    Or = 1,
};

enum class QueryRouteProvider : uint8_t {
    Label = 0,
    RangePosting = 1,
};

enum class QueryRouteExecution : uint8_t {
    None = 0,
    GraphTraversal = 1,
    GpuResidentScan = 2,
    HostCandidateStream = 3,
    GpuResidentScanPostFilter = 4,
};

enum class QueryCandidateSource : uint8_t {
    None = 0,
    LabelPosting = 1,
    RangePosting = 2,
};

constexpr uint32_t query_route_provider_bit(QueryRouteProvider provider)
{
    return 1U << static_cast<uint8_t>(provider);
}

constexpr uint32_t kQueryRouteProviderLabel =
    query_route_provider_bit(QueryRouteProvider::Label);
constexpr uint32_t kQueryRouteProviderRangePosting =
    query_route_provider_bit(QueryRouteProvider::RangePosting);

enum class QueryLabelSourceTier : uint8_t {
    Exact = 0,
    Direct = 1,
    Fallback = 2,
};

enum class QueryManifestOperatorFillMode : uint8_t {
    LegacyFullFallback = 0,
    DirectThenFallback = 1,
    LiteralOnly = 2,
};

enum class OrRouteOrderMode : uint8_t {
    Legacy = 0,
    CostAware = 1,
    CoverageCost = 2,
    RelevanceCoverageCost = 3,
    MarginalAccuracyCost = 4,
};

constexpr uint32_t kOrRouteSketchSidecarVersion = 1;
constexpr uint32_t kOrRouteSketchMinK = 4;
constexpr uint32_t kOrRouteSketchMaxK = 4096;
constexpr uint32_t kOrRouteSketchDefaultK = 64;
constexpr uint64_t kOrRouteSketchDefaultHashSeed = 0xD6E8FEB86659FD93ULL;

struct OrRouteSketchSidecarHeader {
    char magic[16];
    uint32_t version = kOrRouteSketchSidecarVersion;
    uint32_t header_bytes = 224;
    uint32_t endian = 0x01020304;
    uint32_t flags = 1;
    uint64_t token_count = 0;
    uint64_t num_points = 0;
    uint64_t max_row_id_exclusive = 0;
    uint64_t sketch_k = 0;
    uint64_t hash_seed = 0;
    uint64_t hash_count = 0;
    uint64_t group_count = 0;
    uint64_t group_layout_checksum = 0;
    uint64_t source_posting_version = 0;
    uint64_t source_posting_token_count = 0;
    uint64_t source_posting_count = 0;
    uint64_t source_posting_max_row_id_exclusive = 0;
    uint64_t source_posting_file_bytes = 0;
    uint64_t checksum_kind = 0;
    unsigned char source_posting_sha256[32]{};
    unsigned char payload_sha256[32]{};
    char provenance[16]{};
};

static_assert(sizeof(OrRouteSketchSidecarHeader) == 224,
              "OR route sketch sidecar header layout changed");

struct OrRoutePlannerConfig {
    std::string order_mode = "legacy";
    std::string fuzzy_cap_policy = "legacy";
    std::string sketch_sidecar_path;
    std::string accuracy_cost_model_path;
    std::string calibration_trace_path;
    std::string calibration_run_id;
    uint32_t sketch_k = kOrRouteSketchDefaultK;
    uint64_t sketch_hash_seed = kOrRouteSketchDefaultHashSeed;
    bool allow_runtime_sketch_fallback = false;
    uint32_t runtime_sketch_fallback_max_rows = 100000;
    bool verify_sketch_payload_checksum = true;
};

struct RangeAccessPlannerConfig {
    std::string mode = "off";
    std::string and_filter_mode = "auto";
    std::string range_label_and_policy = "auto";
    uint64_t scan_threshold = 0;
    bool drop_wide_hybrid_or = false;
    bool range_graph_hybrid_or_enabled = false;
    bool range_graph_hybrid_and_enabled = false;
};

struct RangeAccessPlannerProfile {
    bool planned = false;
    uint64_t range_posting_routes = 0;
    uint64_t range_posting_rows = 0;
    uint64_t postfilter_routes = 0;
    uint64_t prefilter_routes = 0;
    uint64_t lowfreq_priority_clauses = 0;
    uint64_t label_selected_clauses = 0;
    uint64_t dual_selected_clauses = 0;
    uint64_t range_graph_and_queries = 0;
    uint64_t range_graph_or_queries = 0;
};

const char* query_label_source_tier_name(QueryLabelSourceTier tier);

const char* query_manifest_operator_fill_mode_name(QueryManifestOperatorFillMode mode);

bool is_valid_query_manifest_operator_fill_mode(const std::string& mode);

bool query_manifest_operator_fill_mode_uses_nested_fill(const std::string& mode);

bool query_manifest_operator_fill_mode_requires_operator_index(const std::string& mode);

void validate_query_manifest_operator_rewrite_options(const std::string& fill_mode,
                                                      bool parity_check);

void validate_query_manifest_fuzzy_operator_fill_mode(const std::string& fill_mode);

const char* or_route_order_mode_name(OrRouteOrderMode mode);

bool is_valid_or_route_order_mode(const std::string& mode);

bool is_valid_query_manifest_fuzzy_cap_policy(const std::string& policy);

const char* dense_vector_scalar_type_name(DenseVectorScalarType type);

size_t dense_vector_scalar_size(DenseVectorScalarType type);

bool dense_vector_scalar_is_float(DenseVectorScalarType type);

struct DenseVectorMetadata {
    std::string path;
    DenseVectorScalarType scalar_type = DenseVectorScalarType::Unknown;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct DenseUint8Matrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint8_t> data;

    const uint8_t* row_ptr(uint32_t row) const;
};

struct DenseFloatMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<float> data;

    const float* row_ptr(uint32_t row) const;
};

struct VectorFileInfo {
    std::string path;
    DenseVectorScalarType scalar_type = DenseVectorScalarType::Uint8;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint64_t row_data_offset = 0;
    uint32_t row_stride_bytes = 0;
    uint32_t row_prefix_bytes = 0;
};

struct Int8RerankBuildStats {
    uint64_t rows = 0;
    uint32_t cols = 0;
    uint64_t source_bytes = 0;
    uint64_t artifact_bytes = 0;
};

struct PreparedFloatL2Query {
    const float* values = nullptr;
    uint32_t cols = 0;
    float squared_norm = 0.0f;
};

class RerankVectorStore {
public:
    static std::shared_ptr<RerankVectorStore> map_fp16(
        const VectorFileInfo& source_info);
    static std::shared_ptr<RerankVectorStore> load_memory_int8(
        const std::string& artifact_path,
        const DenseVectorMetadata& source_metadata);

    ~RerankVectorStore();

    RerankVectorStore(const RerankVectorStore&) = delete;
    RerankVectorStore& operator=(const RerankVectorStore&) = delete;

    [[nodiscard]] uint32_t rows() const noexcept;
    [[nodiscard]] uint32_t cols() const noexcept;
    [[nodiscard]] std::string_view mode() const noexcept;
    [[nodiscard]] std::string_view distance_kernel() const noexcept;
    [[nodiscard]] uint64_t mapped_bytes() const noexcept;
    [[nodiscard]] uint64_t resident_payload_bytes() const noexcept;
    [[nodiscard]] PreparedFloatL2Query prepare_l2_query(
        const float* query,
        uint32_t cols) const;
    void read_float_row(uint32_t row, float* dst, uint32_t cols) const;
    [[nodiscard]] float l2_float(uint32_t row,
                                 const PreparedFloatL2Query& query) const;
    [[nodiscard]] float l2_float(uint32_t row,
                                 const float* query,
                                 uint32_t cols) const;
    [[nodiscard]] bool uses_batched_l2() const noexcept;
    void l2_float_batch(const uint32_t* rows, size_t count,
                        const PreparedFloatL2Query& query, float* distances) const;

private:
    struct Impl;
    explicit RerankVectorStore(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

Int8RerankBuildStats build_int8_rerank_artifact(const std::string& source_path,
                                                 const std::string& artifact_path,
                                                 uint32_t row_limit = 0,
                                                 bool force_rebuild = false);

struct SparseLabelMatrix {
    uint32_t rows = 0;
    uint32_t cols = 0;
    std::vector<uint64_t> indptr;
    std::vector<uint32_t> indices;
    // Optional compatibility cache for query-side paths. Data label spmat loads
    // should stay CSR-only to avoid one vector allocation per base row.
    std::vector<std::vector<uint32_t>> row_labels;
    // Optional query-manifest occurrence metadata. When present, both arrays
    // are strictly aligned with indices, including after row sorting.
    std::vector<uint8_t> source_tiers;
    std::vector<uint32_t> relevance_ranks;
};

struct SparseLabelRowSpan {
    const uint32_t* begin = nullptr;
    const uint32_t* end = nullptr;
    bool valid = false;
};

// A clause view borrows generic CSR rows or reconstructs a native two-atom
// clause in stack storage. Copies are safe: pointers never refer to its own arrays.
struct HybridClauseLabels {
    const uint32_t* borrowed_labels = nullptr;
    const uint8_t* borrowed_tiers = nullptr;
    const uint32_t* borrowed_ranks = nullptr;
    std::array<uint32_t, 2> inline_labels{};
    std::array<uint8_t, 2> inline_tiers{};
    std::array<uint32_t, 2> inline_ranks{};
    size_t count = 0;
    bool referenced = false;
    bool valid = false;
    bool has_metadata = false;
    const uint32_t* begin() const { return count ? (referenced ? inline_labels.data() : borrowed_labels) : nullptr; }
    const uint32_t* end() const { return count ? begin() + count : nullptr; }
    uint8_t tier(size_t i) const { return referenced ? inline_tiers[i] : borrowed_tiers ? borrowed_tiers[i] : 0; }
    uint32_t rank(size_t i) const { return referenced ? inline_ranks[i] : borrowed_ranks ? borrowed_ranks[i] : 0; }
};
struct HybridClauseAtomReference {
    uint64_t begin = 0;
    uint32_t count = 0;
    bool range = false;
    bool alternatives = false;
};
struct HybridClauseExpression {
    std::array<HybridClauseAtomReference, 2> atoms{};
    bool active = false;
    bool conjunction = false;
    bool unary = false; // A flat OR of singleton alternatives.
    std::array<uint32_t, 2> branch_ids{};
};

// Compact DNF representation for manifest-driven hybrid queries. Each query
// owns zero or more conjunctive clauses; clauses are unioned by the backend.
// Range-bearing clauses reference the aligned original query range row.
struct HybridQueryClauseTable {
    uint32_t rows = 0;
    std::vector<uint64_t> query_indptr;
    // Per-query logical OR branch budget. Zero means uncapped. Non-hybrid
    // categorical OR rows use query_manifest_max_or_terms here so retention
    // can happen after exact label cardinality is available.
    std::vector<uint32_t> branch_caps;
    // 1 means the cap is the fuzzy semantic-term budget; 0 means it is an
    // explicit categorical OR cap. The two budgets must never compete.
    std::vector<uint8_t> branch_cap_is_fuzzy;
    std::vector<uint64_t> label_indptr;
    std::vector<uint32_t> labels;
    std::vector<uint8_t> source_tiers;
    std::vector<uint32_t> relevance_ranks;
    std::vector<uint32_t> top_branch_ids;
    std::vector<uint8_t> uses_original_range;

    // Original expressions retain all literal matches independently of route
    // selection. No conjunction-label/occurrence payload is expanded per branch.
    std::vector<HybridClauseExpression> expressions;
    std::vector<uint32_t> atom_labels;
    std::vector<uint8_t> atom_source_tiers;
    std::vector<uint32_t> atom_relevance_ranks;

    // Eager descriptors are stored only for unsupported rows. Logical IDs remain
    // global and stable; virtual rows derive descriptors from their expression.
    bool virtual_descriptors = false;
    uint64_t logical_clause_count = 0;
    std::vector<uint64_t> descriptor_indptr;

    size_t clause_count() const { return virtual_descriptors ? logical_clause_count : top_branch_ids.size(); }
    bool descriptor_shape_valid() const {
        const size_t stored = top_branch_ids.size();
        return label_indptr.size() == stored + 1 && uses_original_range.size() == stored &&
            (!virtual_descriptors || (descriptor_indptr.size() == static_cast<size_t>(rows) + 1 &&
                descriptor_indptr.back() == stored && expressions.size() == rows &&
                query_indptr.size() == static_cast<size_t>(rows) + 1 && query_indptr.back() == clause_count()));
    }
    static uint64_t alternatives(const HybridClauseAtomReference& atom) { return atom.alternatives ? atom.count : 1; }
    uint32_t clause_query(uint64_t clause, uint32_t query = UINT32_MAX) const {
        if (clause >= clause_count() || query_indptr.size() != static_cast<size_t>(rows) + 1) return UINT32_MAX;
        if (query == UINT32_MAX) {
            const auto it = std::upper_bound(query_indptr.begin(), query_indptr.end(), clause);
            if (it == query_indptr.begin() || it == query_indptr.end()) return UINT32_MAX;
            query = static_cast<uint32_t>(it - query_indptr.begin() - 1);
        }
        return query < rows && clause >= query_indptr[query] && clause < query_indptr[query + 1] ? query : UINT32_MAX;
    }
    uint64_t descriptor_index(uint64_t clause, uint32_t query) const {
        return virtual_descriptors ? descriptor_indptr[query] + clause - query_indptr[query] : clause;
    }
    uint32_t clause_branch(uint64_t clause, uint32_t query = UINT32_MAX) const {
        query = clause_query(clause, query);
        if (query == UINT32_MAX) return UINT32_MAX;
        if (virtual_descriptors && expressions[query].active) {
            const auto& expression = expressions[query];
            const uint64_t ordinal = clause - query_indptr[query];
            if (expression.unary) return static_cast<uint32_t>(ordinal);
            return expression.conjunction ? 0 : expression.branch_ids[ordinal < alternatives(expression.atoms[0]) ? 0 : 1];
        }
        const auto index = descriptor_index(clause, query);
        return index < top_branch_ids.size() ? top_branch_ids[index] : UINT32_MAX;
    }
    bool clause_uses_range(uint64_t clause, uint32_t query = UINT32_MAX) const {
        query = clause_query(clause, query);
        if (query == UINT32_MAX) return false;
        if (virtual_descriptors && expressions[query].active) {
            const auto& expression = expressions[query];
            if (expression.unary) return false;
            return expression.conjunction ? expression.atoms[0].range || expression.atoms[1].range :
                expression.atoms[clause - query_indptr[query] < alternatives(expression.atoms[0]) ? 0 : 1].range;
        }
        const auto index = descriptor_index(clause, query);
        return index < uses_original_range.size() && uses_original_range[index] != 0;
    }
    void finish_clause_row() {
        if (virtual_descriptors) {
            expressions.resize(static_cast<size_t>(rows) + 1);
            const auto& expression = expressions[rows];
            const uint64_t count = expression.active ? (expression.unary ? alternatives(expression.atoms[0]) :
                expression.conjunction ? alternatives(expression.atoms[0]) * alternatives(expression.atoms[1]) :
                alternatives(expression.atoms[0]) + alternatives(expression.atoms[1])) :
                top_branch_ids.size() - descriptor_indptr.back();
            logical_clause_count += count;
            descriptor_indptr.push_back(top_branch_ids.size());
        }
        query_indptr.push_back(clause_count());
    }
    HybridClauseLabels clause_labels(uint64_t clause, uint32_t query = UINT32_MAX) const
    {
        HybridClauseLabels result;
        if (clause >= clause_count() || !descriptor_shape_valid()) return result;
        if (!expressions.empty()) {
            if (expressions.size() != rows || query_indptr.size() != static_cast<size_t>(rows) + 1) return result;
            if (query == UINT32_MAX) {
                auto it = std::upper_bound(query_indptr.begin(), query_indptr.end(), clause);
                if (it == query_indptr.begin() || it == query_indptr.end()) return result;
                query = static_cast<uint32_t>(it - query_indptr.begin() - 1);
            }
            if (query >= rows || clause < query_indptr[query] || clause >= query_indptr[query + 1]) return result;
            const auto& expression = expressions[query];
            if (expression.active) {
                result.referenced = result.has_metadata = true;
                if (atom_labels.size() != atom_source_tiers.size() || atom_labels.size() != atom_relevance_ranks.size()) return result;
                const auto alternatives = [](const auto& atom) -> uint64_t { return atom.alternatives ? atom.count : 1; };
                const auto append = [&](const HybridClauseAtomReference& atom, uint64_t index) {
                    if (atom.range) return true;
                    if (index >= atom.count || atom.begin > atom_labels.size() || index >= atom_labels.size() - atom.begin || result.count == 2) return false;
                    const size_t i = result.count++, pos = atom.begin + index;
                    result.inline_labels[i] = atom_labels[pos];
                    result.inline_tiers[i] = atom_source_tiers[pos];
                    result.inline_ranks[i] = atom_relevance_ranks[pos];
                    return true;
                };
                const auto& left = expression.atoms[0];
                const auto& right = expression.atoms[1];
                const uint64_t ordinal = clause - query_indptr[query];
                if (expression.unary) {
                    if (!append(left, ordinal)) return result;
                } else if (expression.conjunction) {
                    const uint64_t right_count = alternatives(right);
                    if (!right_count || ordinal >= alternatives(left) * right_count ||
                        !append(left, ordinal / right_count) || !append(right, ordinal % right_count)) return result;
                } else if (ordinal < alternatives(left)) {
                    if (!append(left, ordinal)) return result;
                } else if (!append(right, ordinal - alternatives(left))) return result;
                if (result.count == 2) {
                    const bool swap = result.inline_labels[1] < result.inline_labels[0] ||
                        (result.inline_labels[1] == result.inline_labels[0] &&
                         (result.inline_tiers[1] < result.inline_tiers[0] ||
                          (result.inline_tiers[1] == result.inline_tiers[0] && result.inline_ranks[1] < result.inline_ranks[0])));
                    if (swap) {
                        std::swap(result.inline_labels[0], result.inline_labels[1]);
                        std::swap(result.inline_tiers[0], result.inline_tiers[1]);
                        std::swap(result.inline_ranks[0], result.inline_ranks[1]);
                    }
                    if (result.inline_labels[0] == result.inline_labels[1]) result.count = 1;
                }
                result.valid = true;
                return result;
            }
        }
        const uint32_t owner = virtual_descriptors ? clause_query(clause, query) : 0;
        if (virtual_descriptors && owner == UINT32_MAX) return result;
        const uint64_t stored = virtual_descriptors ? descriptor_index(clause, owner) : clause;
        if (stored >= top_branch_ids.size()) return result;
        const uint64_t begin = label_indptr[stored], end = label_indptr[stored + 1];
        if (begin > end || end > labels.size() ||
            (!source_tiers.empty() && source_tiers.size() != labels.size()) ||
            (!relevance_ranks.empty() && relevance_ranks.size() != labels.size())) return result;
        result.count = end - begin;
        if (result.count) {
            result.borrowed_labels = labels.data() + begin;
            if (!source_tiers.empty()) result.borrowed_tiers = source_tiers.data() + begin;
            if (!relevance_ranks.empty()) result.borrowed_ranks = relevance_ranks.data() + begin;
        }
        result.has_metadata = !source_tiers.empty() || !relevance_ranks.empty();
        result.valid = true;
        return result;
    }
    bool has_clauses(uint32_t query_idx) const
    {
        return query_idx < rows &&
               query_indptr.size() == static_cast<size_t>(rows) + 1 &&
               query_indptr[query_idx] < query_indptr[static_cast<size_t>(query_idx) + 1];
    }
};

inline SparseLabelRowSpan sparse_label_row_span(const SparseLabelMatrix& matrix,
                                                uint32_t row)
{
    if (row >= matrix.rows ||
        matrix.indptr.size() != static_cast<size_t>(matrix.rows) + 1) {
        return {};
    }
    const uint64_t begin64 = matrix.indptr[row];
    const uint64_t end64 = matrix.indptr[static_cast<size_t>(row) + 1];
    if (begin64 > end64 || end64 > matrix.indices.size()) {
        return {};
    }
    if (begin64 == end64) {
        return {nullptr, nullptr, true};
    }
    const size_t begin = static_cast<size_t>(begin64);
    const size_t end = static_cast<size_t>(end64);
    return {matrix.indices.data() + begin, matrix.indices.data() + end, true};
}

inline bool sample_matches_query_labels_binary(const SparseLabelMatrix& data_labels,
                                               uint32_t sample_id,
                                               const SparseLabelMatrix& query_labels_matrix,
                                               uint32_t query_idx,
                                               uint32_t primary_label) {
    const SparseLabelRowSpan sample_labels = sparse_label_row_span(data_labels, sample_id);
    const SparseLabelRowSpan query_labels = sparse_label_row_span(query_labels_matrix, query_idx);
    if (!sample_labels.valid || !query_labels.valid) {
        return false;
    }
    const bool sample_is_empty = sample_labels.begin == sample_labels.end;
    for (const uint32_t* it = query_labels.begin; it != query_labels.end; ++it) {
        const uint32_t label = *it;
        if (label == primary_label) continue;
        if (sample_is_empty ||
            !std::binary_search(sample_labels.begin, sample_labels.end, label)) {
            return false;
        }
    }
    return true;
}

struct Uint32MatrixIbin {
    uint64_t rows = 0;
    uint64_t cols = 0;
    std::vector<uint32_t> data;
};

struct QueryRangeTable {
    uint32_t rows = 0;
    uint32_t range_count = 1;
    std::vector<uint8_t> modes;
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;

    size_t flat_index(uint32_t row, uint32_t range_idx) const;
};

struct RangeLabelIntervals {
    uint32_t range_count = 1;
    std::vector<uint8_t> has_interval;
    std::vector<float> lower_bounds;
    std::vector<float> upper_bounds;
    std::vector<uint32_t> level_indices;
    std::vector<uint32_t> bin_indices;
    std::vector<uint32_t> range_indices;
};

void coarsen_range_cover_bins(std::vector<uint8_t>& child_active,
                              std::vector<uint8_t>& parent_active,
                              const std::vector<uint32_t>& parent_labels);

struct LowfreqPostingIndex {
    uint32_t group_count = 0;
    uint32_t total_rows = 0;
    std::vector<uint32_t> label_size;
    std::vector<uint32_t> label_offset;
    std::vector<uint32_t> sample_ids;
    std::vector<uint64_t> row_filter_mask;
};

struct GroupLayout {
    std::vector<std::vector<uint32_t>> groups;
    std::vector<uint32_t> label_to_group;
};

struct GraphMetadata {
    uint32_t label_count = 0;
    uint32_t original_label_count = 0;
    uint32_t total_rows = 0;
    int specificity_threshold = 2000;
    std::vector<uint32_t> cat_freq;
    std::vector<uint32_t> group_posting_size;
    std::vector<uint32_t> label_size;
    std::vector<uint32_t> label_offset;
    std::vector<uint32_t> row_to_sample;
    std::vector<uint64_t> row_filter_mask;
    std::vector<uint64_t> label_filter_bit;
    // Identity range graphs have one label per group, so their group-local
    // row mask is semantically all ones. Keep that representation implicit
    // instead of materializing one uint64_t per graph row.
    bool implicit_all_ones_row_filter_mask = false;
    // Exact base-point count for each original label. For grouped indexes this
    // is derived from the group-local row masks, never from the group union size.
    std::vector<uint32_t> original_label_cardinality;
};

struct GraphBuildIndexes {
    GraphMetadata metadata;
    LowfreqPostingIndex lowfreq_index;
};

struct GraphSeedFileHeader {
    uint32_t version = kGraphSeedVersion;
    uint32_t reserved = 0;
    uint64_t label_count = 0;
    uint64_t num_points = 0;
    uint64_t graph_rows = 0;
    uint64_t graph_degree = 0;
    uint64_t specificity_threshold = 0;
};

struct GraphOriginalLabelSeedHeader {
    uint32_t version = 2;
    uint32_t reserved = 0;
    uint64_t original_label_count = 0;
    uint64_t group_count = 0;
    uint64_t num_points = 0;
    uint64_t graph_rows = 0;
    uint64_t graph_degree = 0;
    uint64_t specificity_threshold = 0;
};

struct GraphMetadataSidecarHeader {
    uint32_t version = kGraphMetadataSidecarVersion;
    uint32_t reserved = 0;
    uint64_t label_count = 0;
    uint64_t group_count = 0;
    uint64_t original_label_count = 0;
    uint64_t num_points = 0;
    uint64_t row_limit = 0;
    uint64_t graph_rows = 0;
    uint64_t total_rows = 0;
    // Legacy/reserved: written for old readers, but not a metadata compatibility key.
    uint64_t graph_degree = 0;
    uint64_t specificity_threshold = 0;
    uint64_t group_layout_checksum = 0;
};

struct OrRoutePlannerProfile {
    OrRouteOrderMode mode = OrRouteOrderMode::Legacy;
    uint32_t total_cap = 0;
    uint32_t highfreq_cap = 0;
    uint64_t queries = 0;
    uint64_t original_routes = 0;
    uint64_t emitted_routes = 0;
    // Post-selection physical sharing; emitted_routes remains the bounded clause-plan count.
    uint64_t same_producer_merged_routes = 0;
    uint64_t original_highfreq_routes = 0;
    uint64_t original_lowfreq_routes = 0;
    uint64_t emitted_highfreq_routes = 0;
    uint64_t emitted_lowfreq_routes = 0;
    uint64_t original_exact_source_routes = 0;
    uint64_t emitted_exact_source_routes = 0;
    uint64_t original_direct_source_routes = 0;
    uint64_t emitted_direct_source_routes = 0;
    uint64_t original_fallback_source_routes = 0;
    uint64_t emitted_fallback_source_routes = 0;
    uint64_t retained_relevance_rank_sum = 0;
    uint32_t retained_relevance_rank_max = 0;
    uint64_t reordered_queries = 0;
    uint64_t reordered_routes = 0;
    uint64_t capped_queries = 0;
    uint64_t capped_routes = 0;
    uint64_t total_cap_capped_routes = 0;
    uint64_t highfreq_cap_capped_routes = 0;
    uint64_t manifest_cap_queries = 0;
    uint32_t manifest_cap_max = 0;
    uint64_t original_exact_lowfreq_rows = 0;
    uint64_t emitted_exact_lowfreq_rows = 0;
    uint64_t original_highfreq_cost_proxy = 0;
    uint64_t emitted_highfreq_cost_proxy = 0;
    // Sum of per-route exact label cardinalities. Route overlaps are counted
    // repeatedly; these fields are not an OR-union cardinality.
    uint64_t original_exact_label_cardinality_sum = 0;
    uint64_t emitted_exact_label_cardinality_sum = 0;
    uint64_t sketch_labels = 0;
    uint64_t sketch_hashes = 0;
    uint64_t sketch_bytes = 0;
    uint64_t sketch_sidecar_bytes_read = 0;
    uint64_t sketch_build_rows = 0;
    uint64_t retained_zero_marginal_routes = 0;
    // Set when exact cardinality was used for either a configured OR mode or
    // a manifest-provided logical branch cap.
    bool exact_cardinality_used = false;
    double estimated_union_coverage_sum = 0.0;
    double retained_estimated_marginal_gain_sum = 0.0;
    double original_predicted_latency_us_sum = 0.0;
    double emitted_predicted_latency_us_sum = 0.0;
    double emitted_quality_sum = 0.0;
    double emitted_utility_sum = 0.0;
    std::string calibration_id;
};

// Wall-clock accounting for work performed after a query workload has been
// selected but before the backend starts its formal QPS window. Runtime totals
// contain the component stages. Manifest detail fields are nested diagnostics
// for query_manifest_prepare_ms; route_generation_ms excludes cardinality init.
struct GraphQueryPreparationTiming {
    double runtime_load_total_ms = 0.0;
    double query_vector_inspect_ms = 0.0;
    double query_vector_read_ms = 0.0;
    double query_label_spmat_read_ms = 0.0;
    double query_label_materialize_ms = 0.0;
    double query_manifest_prepare_ms = 0.0;
    double query_manifest_file_read_detail_ms = 0.0;
    double query_manifest_json_parse_detail_ms = 0.0;
    double query_manifest_record_select_detail_ms = 0.0;
    double query_manifest_exact_extract_detail_ms = 0.0;
    double query_manifest_fuzzy_expand_wall_detail_ms = 0.0;
    double query_manifest_row_materialize_detail_ms = 0.0;
    double query_range_read_ms = 0.0;
    double route_generation_ms = 0.0;
    // Nested, disjoint wall spans within route_generation_ms.
    double route_select_emit_ms = 0.0;
    double route_union_ms = 0.0;
    double route_payload_materialize_ms = 0.0;
    double static_label_cardinality_init_ms = 0.0;
    double range_access_plan_ms = 0.0;
    double prefetch_build_ms = 0.0;
    double host_candidate_label_load_ms = 0.0;
    double static_manifest_discovery_ms = 0.0;
    double static_manifest_ngram_load_ms = 0.0;
    double static_manifest_operator_load_ms = 0.0;
    double static_manifest_fallback_index_build_ms = 0.0;
    double static_range_posting_index_load_ms = 0.0;
    std::map<std::string, double> static_stages_ms;

    double query_stage_sum_ms() const
    {
        return query_vector_inspect_ms + query_vector_read_ms +
            query_label_spmat_read_ms + query_label_materialize_ms +
            query_manifest_prepare_ms + query_range_read_ms +
            route_generation_ms + range_access_plan_ms;
    }
};

// Transient online planning state. Execution payloads stay empty until provider
// selection and physical-route union finish. Never publish a deferred state.
struct QueryRoutePayloadPreparation {
    bool deferred = false;
    // bit 0: no label residual; bit 1: suppress the flat range prefilter.
    std::vector<uint8_t> flags;
    std::vector<uint32_t> excluded_labels;
};

struct GraphRuntime {
    DenseVectorScalarType vector_scalar_type = DenseVectorScalarType::Uint8;
    DenseVectorMetadata base_vector_metadata;
    QueryRoutePayloadPreparation route_payload_preparation;
    // Native consumers resolve route IDs through the original-query table.
    // Shape-only routed matrices retain route counts; their payloads are empty.
    bool shared_route_query_vectors = false;
    DenseVectorMetadata query_vector_metadata;
    DenseUint8Matrix base_vectors;
    DenseFloatMatrix base_float_vectors;
    VectorFileInfo base_vector_file;
    std::shared_ptr<RerankVectorStore> rerank_vector_store;
    std::shared_ptr<go_fann::range_posting::Index> range_posting_index;
    std::string rerank_vector_storage_mode = "default";
    std::string rerank_vector_artifact_path;
    uint32_t base_range_count = 1;
    std::vector<float> base_range_values;
    DenseUint8Matrix original_query_vectors;
    DenseUint8Matrix query_vectors;
    DenseFloatMatrix original_query_float_vectors;
    DenseFloatMatrix query_float_vectors;
    uint32_t original_query_count = 0;
    SparseLabelMatrix data_labels;
    SparseLabelMatrix original_query_labels_matrix;
    std::vector<QueryLabelOp> original_query_label_ops;
    HybridQueryClauseTable original_query_clauses;
    SparseLabelMatrix query_labels_matrix;
    QueryRangeTable original_query_ranges;
    QueryRangeTable query_ranges;
    Uint32MatrixIbin graph;
    LowfreqPostingIndex lowfreq_index;
    GroupLayout group_layout;
    GraphMetadata metadata;
    RangeLabelIntervals label_intervals;
    std::vector<uint32_t> seed_local_id_by_label;   // key: group_id, value: local_row (group-based seed selection)
    std::vector<uint32_t> seed_by_original_label;   // key: original_label, value: local_row within its group (label-based seed selection)
    std::vector<uint32_t> query_group_ids;          // group_id for each routed query
    std::vector<uint32_t> query_filter_labels;      // original_label for filter
    std::vector<uint64_t> query_filter_bits;
    std::vector<uint32_t> start_local_ids;
    std::vector<uint8_t> query_is_lowfreq;
    std::vector<uint32_t> route_to_original_query;
    std::vector<uint32_t> route_to_clause;
    std::vector<uint64_t> route_clause_indptr;
    std::vector<uint32_t> route_clause_ids;
    std::vector<uint32_t> route_top_branch_ids;
    std::vector<uint8_t> route_enabled;
    std::vector<QueryRouteProvider> route_providers;
    std::vector<QueryRouteExecution> route_execution_modes;
    std::vector<QueryCandidateSource> route_candidate_sources;
    std::vector<uint64_t> route_range_posting_begins;
    std::vector<uint64_t> route_range_posting_sizes;
    std::vector<uint32_t> route_range_posting_indices;
    std::vector<uint8_t> range_graph_and_original_queries;
    std::vector<uint8_t> range_graph_or_original_queries;
    std::vector<uint64_t> range_graph_clause_indptr;
    std::vector<uint32_t> range_graph_clause_ids;
    std::vector<uint32_t> high_query_indices;
    std::vector<uint32_t> low_query_indices;
    std::vector<Node> prefetch_nodes;
    std::vector<uint32_t> prefetch_index;
    std::vector<uint32_t> prefetch_begin_by_group;
    std::vector<uint32_t> prefetch_end_by_group;
    OrRoutePlannerProfile or_route_planner_profile;
    RangeAccessPlannerProfile range_access_planner_profile;
    GraphQueryPreparationTiming query_preparation_timing;
    std::string seed_selection_mode = "group";
    double and_dual_route_posting_ratio_threshold =
        go_fann::defaults::default_and_dual_route_posting_ratio_threshold;
    uint32_t supported_query_route_providers = kQueryRouteProviderLabel;

    const uint8_t* route_query_u8(uint32_t route) const
    {
        return shared_route_query_vectors
            ? original_query_vectors.row_ptr(route_to_original_query[route])
            : query_vectors.row_ptr(route);
    }
    const float* route_query_float(uint32_t route) const
    {
        return shared_route_query_vectors
            ? original_query_float_vectors.row_ptr(route_to_original_query[route])
            : query_float_vectors.row_ptr(route);
    }
};

struct GraphQueryState {
    QueryRoutePayloadPreparation route_payload_preparation;
    bool shared_route_query_vectors = false;
    DenseVectorMetadata query_vector_metadata;
    DenseUint8Matrix original_query_vectors;
    DenseUint8Matrix query_vectors;
    DenseFloatMatrix original_query_float_vectors;
    DenseFloatMatrix query_float_vectors;
    uint32_t original_query_count = 0;
    SparseLabelMatrix original_query_labels_matrix;
    std::vector<QueryLabelOp> original_query_label_ops;
    HybridQueryClauseTable original_query_clauses;
    SparseLabelMatrix query_labels_matrix;
    QueryRangeTable original_query_ranges;
    QueryRangeTable query_ranges;
    std::vector<uint32_t> query_group_ids;
    std::vector<uint32_t> query_filter_labels;
    std::vector<uint64_t> query_filter_bits;
    std::vector<uint32_t> start_local_ids;
    std::vector<uint8_t> query_is_lowfreq;
    std::vector<uint32_t> route_to_original_query;
    std::vector<uint32_t> route_to_clause;
    std::vector<uint64_t> route_clause_indptr;
    std::vector<uint32_t> route_clause_ids;
    std::vector<uint32_t> route_top_branch_ids;
    std::vector<uint8_t> route_enabled;
    std::vector<QueryRouteProvider> route_providers;
    std::vector<QueryRouteExecution> route_execution_modes;
    std::vector<QueryCandidateSource> route_candidate_sources;
    std::vector<uint64_t> route_range_posting_begins;
    std::vector<uint64_t> route_range_posting_sizes;
    std::vector<uint32_t> route_range_posting_indices;
    std::vector<uint8_t> range_graph_and_original_queries;
    std::vector<uint8_t> range_graph_or_original_queries;
    std::vector<uint64_t> range_graph_clause_indptr;
    std::vector<uint32_t> range_graph_clause_ids;
    std::vector<uint32_t> high_query_indices;
    std::vector<uint32_t> low_query_indices;
    OrRoutePlannerProfile or_route_planner_profile;
    RangeAccessPlannerProfile range_access_planner_profile;
    GraphQueryPreparationTiming query_preparation_timing;
};

inline bool sample_matches_clause_labels(const GraphRuntime& runtime,
                                         uint32_t clause_idx,
                                         uint32_t sample_id)
{
    const HybridQueryClauseTable& clauses = runtime.original_query_clauses;
    if (clause_idx >= clauses.clause_count() || sample_id >= runtime.data_labels.rows ||
        !clauses.descriptor_shape_valid()) {
        return false;
    }
    const SparseLabelRowSpan sample_labels = sparse_label_row_span(runtime.data_labels,
                                                                   sample_id);
    if (!sample_labels.valid) {
        return false;
    }
    const auto clause_labels = clauses.clause_labels(clause_idx);
    if (!clause_labels.valid) return false;
    for (const uint32_t* it = clause_labels.begin(); it != clause_labels.end(); ++it) {
        const uint32_t label = *it;
        if (sample_labels.begin == sample_labels.end ||
            !std::binary_search(sample_labels.begin, sample_labels.end, label)) {
            return false;
        }
    }
    return true;
}

inline bool sample_matches_hybrid_range_branch_labels(const GraphRuntime& runtime,
                                                      uint32_t original_query_idx,
                                                      uint32_t sample_id)
{
    const HybridQueryClauseTable& clauses = runtime.original_query_clauses;
    if (!clauses.has_clauses(original_query_idx)) {
        return false;
    }
    const uint64_t clause_begin = clauses.query_indptr[original_query_idx];
    const uint64_t clause_end = clauses.query_indptr[original_query_idx + 1];
    if (clause_begin > clause_end || clause_end > clauses.clause_count()) {
        return false;
    }
    for (uint64_t clause_idx = clause_begin; clause_idx < clause_end; ++clause_idx) {
        if (clauses.clause_uses_range(clause_idx, original_query_idx) &&
            sample_matches_clause_labels(
                runtime, static_cast<uint32_t>(clause_idx), sample_id)) {
            return true;
        }
    }
    return false;
}

inline bool sample_matches_route_clause_labels(const GraphRuntime& runtime,
                                               uint32_t route_idx,
                                               uint32_t sample_id)
{
    if (route_idx < runtime.route_to_original_query.size() &&
        runtime.route_clause_indptr.size() == runtime.route_to_original_query.size() + 1) {
        const uint64_t begin = runtime.route_clause_indptr[route_idx];
        const uint64_t end = runtime.route_clause_indptr[route_idx + 1];
        if (begin > end || end > runtime.route_clause_ids.size()) {
            return false;
        }
        if (begin != end) {
            for (uint64_t idx = begin; idx < end; ++idx) {
                if (sample_matches_clause_labels(
                        runtime, runtime.route_clause_ids[idx], sample_id)) {
                    return true;
                }
            }
            return false;
        }
    }
    if (route_idx >= runtime.query_filter_labels.size()) {
        return false;
    }
    return sample_matches_query_labels_binary(runtime.data_labels,
                                              sample_id,
                                              runtime.query_labels_matrix,
                                              route_idx,
                                              runtime.query_filter_labels[route_idx]);
}

inline bool sample_matches_range_graph_clause_labels(const GraphRuntime& runtime,
                                                     uint32_t original_query_idx,
                                                     uint32_t sample_id)
{
    if (original_query_idx >= runtime.original_query_count ||
        runtime.range_graph_clause_indptr.size() != runtime.original_query_count + 1) {
        return false;
    }
    const uint64_t begin = runtime.range_graph_clause_indptr[original_query_idx];
    const uint64_t end = runtime.range_graph_clause_indptr[original_query_idx + 1];
    if (begin > end || end > runtime.range_graph_clause_ids.size()) {
        return false;
    }
    for (uint64_t idx = begin; idx < end; ++idx) {
        if (sample_matches_clause_labels(
                runtime, runtime.range_graph_clause_ids[idx], sample_id)) {
            return true;
        }
    }
    return false;
}

// Exact owned DNF, including clause-scoped numeric conditions. Ordinary routes
// retain their routed conjunction/range semantics.
bool sample_matches_route_predicate(const GraphRuntime& runtime,
                                    uint32_t route_idx, uint32_t sample_id);

inline bool sample_matches_lowfreq_full_labels(const GraphRuntime& runtime,
                                               uint32_t query_idx,
                                               uint32_t sample_id)
{
    return sample_matches_route_predicate(runtime, query_idx, sample_id);
}

struct GraphRouteSplitStats {
    size_t highfreq_route_count = 0;
    size_t lowfreq_route_count = 0;
    size_t provider_route_count = 0;
    size_t unroutable_route_count = 0;
    int64_t highfreq_total_steps = 0;
    int64_t lowfreq_total_steps = 0;
};

struct GraphLoadConfig {
    std::string graph_path;
    std::string graph_storage_mode = "memory";
    std::string rerank_vector_storage_mode = "default";
    std::string rerank_vector_artifact_path;
    // Secondary graph runtimes borrow the primary runtime's vector store after
    // loading. They still inspect vector metadata, but must not map the same
    // backing file during construction.
    bool defer_rerank_vector_store = false;
    bool allow_float_vectors = false;
    bool highfreq_prefetch_with_vector_payload = true;
    bool highfreq_prefetch_with_sample_ids = false;
    bool load_data_labels_for_filtering = false;
    std::string data_label_spmat_path;
    std::string query_label_spmat_path;
    // A secondary range graph can reuse the primary query vectors/ranges while
    // deliberately discarding label predicates.  This produces only virtual
    // range-label routes and avoids materializing a synthetic empty .spmat.
    bool force_empty_query_labels = false;
    bool implicit_all_ones_row_filter_mask = false;
    // When non-empty, retain global query numbering but build physical routes
    // only for selected original queries. This lets another graph consume a
    // subset while sharing the primary graph's query preprocessing state.
    std::vector<uint8_t> original_query_route_mask;
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
    std::string seed_sidecar_path;
    std::string original_label_seed_sidecar_path;
    std::string metadata_sidecar_path;
    std::string query_range_path;
    std::string range_label_manifest_path;
    std::string range_posting_index_path;
    RangeAccessPlannerConfig range_access_planner;
    std::string vector_consumer = "graph runtime";
    // Backend capability: consumers must evaluate owned residual DNF.
    bool supports_shared_label_routes = false;
    // Enable only when every backend query consumer uses route_query_* or
    // the original-query matrix. Legacy/PQ dense uploads retain their adapter.
    bool supports_shared_query_vectors = false;
    std::string group_path;
    std::string lowfreq_index_path;
    std::string routing_mode = "greedy_min_posting";
    std::string seed_selection_mode = "group";
    uint32_t dataset_cutoff = 0;
    int specificity_threshold = 2000;
    int prefetch_hops = 3;
    int dim = 128;
    int degree = 64;
    double and_dual_route_posting_ratio_threshold =
        go_fann::defaults::default_and_dual_route_posting_ratio_threshold;
    uint32_t supported_query_route_providers = kQueryRouteProviderLabel;
};

void configure_query_manifest_operator_rewrite(const std::string& operator_index_path,
                                               bool parity_check,
                                               const std::string& fill_mode =
                                                   "legacy_full_fallback");

void configure_or_route_planner(const std::string& order_mode);

void configure_or_route_planner(const OrRoutePlannerConfig& config);

struct FuzzyOperatorRewriteTestResult {
    std::vector<uint32_t> operator_labels;
    std::vector<uint8_t> operator_source_tiers;
    std::vector<uint32_t> operator_relevance_ranks;
    std::vector<uint32_t> baseline_labels;
    bool used_fallback = false;
    uint64_t direct_candidates = 0;
    uint64_t direct_matches = 0;
};

struct FuzzyManifestRewriteTestResult {
    std::vector<uint32_t> labels;
    std::vector<uint8_t> source_tiers;
    std::vector<uint32_t> relevance_ranks;
};

struct HybridManifestParseTestResult {
    SparseLabelMatrix union_labels;
    HybridQueryClauseTable clauses;
    QueryLabelOp top_op = QueryLabelOp::And;
};

HybridManifestParseTestResult evaluate_hybrid_query_manifest_row_for_test(
    const std::string& json_line,
    const std::vector<uint32_t>& fuzzy_labels,
    const std::vector<uint8_t>& fuzzy_source_tiers,
    const std::vector<uint32_t>& fuzzy_relevance_ranks,
    uint32_t label_count, bool lazy_clauses = false);

FuzzyOperatorRewriteTestResult evaluate_fuzzy_operator_rewrite_for_test(
    const std::string& token_ngram_index_path,
    const std::string& operator_index_path,
    const std::string& pattern,
    const std::string& match_mode,
    float min_score,
    uint32_t max_terms,
    const std::string& fill_mode = "legacy_full_fallback");

FuzzyManifestRewriteTestResult evaluate_fuzzy_operator_manifest_row_for_test(
    const std::string& token_ngram_index_path,
    const std::string& operator_index_path,
    const std::vector<std::string>& patterns,
    const std::vector<std::string>& match_modes,
    float min_score,
    uint32_t max_terms,
    const std::string& fill_mode = "direct_then_fallback");

void exercise_fuzzy_operator_rewrite_concurrently_for_test(
    const std::string& token_ngram_index_path,
    const std::string& operator_index_path,
    const std::string& pattern,
    const std::string& match_mode,
    float min_score,
    uint32_t max_terms,
    uint32_t thread_count,
    uint32_t repetitions_per_thread,
    const std::string& fill_mode = "legacy_full_fallback");

std::string lower_extension(const std::string& path);

DenseVectorMetadata inspect_dense_vector_file(const std::string& path, uint32_t row_limit = 0);

DenseUint8Matrix load_uint8_ibin_like(const std::string& path, uint32_t row_limit = 0);

DenseUint8Matrix load_bvecs_matrix(const std::string& path, uint32_t row_limit = 0);

DenseUint8Matrix load_uint8_matrix_auto(const std::string& path, uint32_t row_limit = 0);

DenseFloatMatrix load_fvecs_matrix(const std::string& path, uint32_t row_limit = 0);

DenseFloatMatrix load_float_matrix_auto(const std::string& path, uint32_t row_limit = 0);

VectorFileInfo inspect_uint8_vector_file(const std::string& path, uint32_t row_limit = 0);
VectorFileInfo inspect_vector_file(const std::string& path, uint32_t row_limit = 0);

void require_uint8_vector_file(const DenseVectorMetadata& metadata,
                               const std::string& context);

void require_uint8_graph_runtime(const GraphRuntime& runtime,
                                   const std::string& context);

void copy_base_vector_row(const GraphRuntime& runtime,
                          uint32_t sample_id,
                          uint8_t* dst,
                          uint32_t cols);

void copy_lowfreq_base_vector_row(const GraphRuntime& runtime,
                                  uint32_t sample_id,
                                  uint8_t* dst,
                                  uint32_t cols);
void copy_base_float_vector_row(const GraphRuntime& runtime,
                                uint32_t sample_id,
                                float* dst,
                                uint32_t cols);
PreparedFloatL2Query prepare_base_float_vector_l2_query(
    const GraphRuntime& runtime,
    const float* query,
    uint32_t cols);
float compute_base_float_vector_l2(const GraphRuntime& runtime,
                                   uint32_t sample_id,
                                   const PreparedFloatL2Query& query);
void copy_lowfreq_base_float_vector_row(const GraphRuntime& runtime,
                                        uint32_t sample_id,
                                        float* dst,
                                        uint32_t cols);

std::vector<float> load_float_fbin_column(const std::string& path,
                                          uint32_t row_limit = 0,
                                          uint32_t* out_rows = nullptr,
                                          uint32_t* out_cols = nullptr);

SparseLabelMatrix load_spmat_labels(const std::string& path,
                                    uint32_t row_limit = 0,
                                    bool materialize_row_labels = false);

QueryRangeTable load_query_range_binary(const std::string& path, uint32_t expected_query_count);

RangeLabelIntervals load_range_label_manifest(const std::string& path, uint32_t expected_label_count);

Uint32MatrixIbin load_uint32_matrix_ibin(const std::string& path);
Uint32MatrixIbin load_graph_matrix_auto(const std::string& path);
LowfreqPostingIndex load_lowfreq_posting_index(const std::string& path);
void save_lowfreq_posting_index(const std::string& path, const LowfreqPostingIndex& index);
LowfreqPostingIndex build_lowfreq_posting_index(const SparseLabelMatrix& data_labels,
                                                const GroupLayout& group_layout,
                                                int specificity_threshold);

GroupLayout build_identity_group_layout(uint32_t label_count);

GroupLayout load_group_layout(const std::string& path, uint32_t label_count);

GraphMetadata rebuild_graph_metadata(const SparseLabelMatrix& data_labels,
                                         int specificity_threshold);

GraphMetadata rebuild_graph_metadata(const SparseLabelMatrix& data_labels,
                                         int specificity_threshold,
                                         const GroupLayout& group_layout);

GraphBuildIndexes build_graph_build_indexes(const SparseLabelMatrix& data_labels,
                                                           int specificity_threshold,
                                                           const GroupLayout& group_layout);

std::vector<uint32_t> build_seed_local_ids(const DenseUint8Matrix& base_vectors,
                                           const GraphMetadata& metadata,
                                           int dim);
std::vector<uint32_t> build_seed_local_ids(const DenseFloatMatrix& base_vectors,
                                           const GraphMetadata& metadata,
                                           int dim);

std::vector<uint32_t> build_seeds_by_original_label(const DenseUint8Matrix& base_vectors,
                                                    const GraphMetadata& metadata,
                                                    const GroupLayout& group_layout,
                                                    int dim);
std::vector<uint32_t> build_seeds_by_original_label(const DenseFloatMatrix& base_vectors,
                                                    const GraphMetadata& metadata,
                                                    const GroupLayout& group_layout,
                                                    int dim);

bool load_seed_sidecar(const std::string& path,
                       const GraphMetadata& metadata,
                       uint64_t num_points,
                       uint64_t graph_degree,
                       int specificity_threshold,
                       std::vector<uint32_t>& seed_local_id_by_label);

void save_seed_sidecar(const std::string& path,
                       const GraphMetadata& metadata,
                       uint64_t num_points,
                       uint64_t graph_degree,
                       int specificity_threshold,
                       const std::vector<uint32_t>& seed_local_id_by_label);

bool load_original_label_seed_sidecar(const std::string& path,
                                      const GraphMetadata& metadata,
                                      const GroupLayout& group_layout,
                                      uint64_t num_points,
                                      uint64_t graph_degree,
                                      int specificity_threshold,
                                      std::vector<uint32_t>& seed_by_original_label);

void save_original_label_seed_sidecar(const std::string& path,
                                      const GraphMetadata& metadata,
                                      const GroupLayout& group_layout,
                                      uint64_t num_points,
                                      uint64_t graph_degree,
                                      int specificity_threshold,
                                      const std::vector<uint32_t>& seed_by_original_label);

void save_metadata_sidecar(const std::string& path,
                           const GraphMetadata& metadata,
                           uint64_t num_points,
                           uint64_t row_limit,
                           uint64_t graph_rows,
                           uint64_t graph_degree,
                           int specificity_threshold,
                           const GroupLayout& group_layout);

// Configure once before loading a workload. All online preparation is included
// in its existing stage timer; no query routes are persisted between loads.
struct QueryPreparationConfig {
    uint32_t threads = 1;  // Explicit 0: automatic, at most eight workers.
    bool optimized = true;
};
void configure_query_preparation(const QueryPreparationConfig& config);

void validate_and_prepare_queries(
    GraphRuntime& runtime,
    int dim,
    const std::string& routing_mode,
    const std::vector<uint8_t>& original_query_route_mask = {},
    bool defer_payloads = false);

// Finish deferred preparation exactly once, after provider choice and union.
void materialize_graph_route_payloads(GraphRuntime& runtime);

void plan_graph_range_access(
    GraphRuntime& runtime,
    const go_fann::range_posting::Index& posting_index,
    const RangeAccessPlannerConfig& config);

// Run once after bounded clause selection AND range-provider selection.
// Returns duplicate physical routes merged; preserves prior disabled mappings.
size_t coalesce_graph_routes(GraphRuntime& runtime);

void validate_graph_runtime_query_metadata(const GraphRuntime& runtime,
                                            const std::string& context);

// Retain routes whose original-query id is set in `selected_original_queries`.
// The original query matrices remain intact so multiple execution contexts can
// share the same original-query GPU preprocessing state.
size_t retain_graph_routes_for_original_queries(
    GraphRuntime& runtime,
    const std::vector<uint8_t>& selected_original_queries);

GraphRouteSplitStats collect_graph_route_split_stats(const GraphRuntime& runtime,
                                                         const std::vector<int>& route_steps);

std::vector<int> build_graph_original_route_flags(const GraphRuntime& runtime);

GraphRuntime load_graph_runtime(const GraphLoadConfig& config);

GraphQueryState load_graph_query_state(GraphRuntime& runtime,
                                           const GraphLoadConfig& config);

// Host-only online planning entry. Its private runtime contains index metadata,
// never graph adjacency, base vectors or posting sample IDs, and cannot be
// passed to an execution backend. Grouped catalogs require explicit exact
// cardinalities; they never derive them by scanning base rows during a query.
class GraphRoutingCatalog {
    GraphRuntime metadata_;
    friend GraphRoutingCatalog load_graph_routing_catalog(
        const GraphLoadConfig&, const std::vector<uint32_t>&);
    friend GraphQueryState prepare_graph_query_batch(
        GraphRoutingCatalog&, const GraphLoadConfig&);
};
GraphRoutingCatalog load_graph_routing_catalog(
    const GraphLoadConfig& config,
    const std::vector<uint32_t>& exact_label_cardinalities = {});
GraphQueryState prepare_graph_query_batch(
    GraphRoutingCatalog& catalog, const GraphLoadConfig& config);

void swap_graph_query_state(GraphRuntime& runtime,
                              GraphQueryState& query_state) noexcept;
