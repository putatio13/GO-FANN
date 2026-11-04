#include "app/config_loader.h"
#include "data/dataloader.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws_contains(Function&& function,
                             const std::string& expected,
                             const std::string& message)
{
    try {
        function();
    } catch (const std::exception& error) {
        require(std::string(error.what()).find(expected) != std::string::npos,
                message + ": " + error.what());
        return;
    }
    throw std::runtime_error(message);
}

uint8_t tier(QueryLabelSourceTier source_tier)
{
    return static_cast<uint8_t>(source_tier);
}

class ScopedRouteCaps {
public:
    ScopedRouteCaps()
        : old_total_(read("GO_FANN_OR_ROUTE_TOTAL_CAP")),
          old_high_(read("GO_FANN_OR_HIGHFREQ_CAP"))
    {
        clear();
    }

    ~ScopedRouteCaps()
    {
        restore("GO_FANN_OR_ROUTE_TOTAL_CAP", old_total_);
        restore("GO_FANN_OR_HIGHFREQ_CAP", old_high_);
    }

    void clear() const
    {
        unsetenv("GO_FANN_OR_ROUTE_TOTAL_CAP");
        unsetenv("GO_FANN_OR_HIGHFREQ_CAP");
    }

    void set(uint32_t total, uint32_t high) const
    {
        setenv("GO_FANN_OR_ROUTE_TOTAL_CAP", std::to_string(total).c_str(), 1);
        setenv("GO_FANN_OR_HIGHFREQ_CAP", std::to_string(high).c_str(), 1);
    }

private:
    static std::optional<std::string> read(const char* name)
    {
        const char* value = std::getenv(name);
        return value == nullptr ? std::nullopt : std::optional<std::string>(value);
    }

    static void restore(const char* name, const std::optional<std::string>& value)
    {
        if (value.has_value()) {
            setenv(name, value->c_str(), 1);
        } else {
            unsetenv(name);
        }
    }

    std::optional<std::string> old_total_;
    std::optional<std::string> old_high_;
};

GraphRuntime make_runtime(const std::vector<uint32_t>& labels,
                            QueryLabelOp op,
                            const std::vector<uint32_t>& group_posting_rows,
                            const std::vector<uint32_t>& graph_rows,
                            const std::vector<uint32_t>& lowfreq_rows,
                            uint64_t graph_degree = 4,
                            const std::vector<uint32_t>& original_label_cardinality = {},
                            const std::vector<uint8_t>& source_tiers = {},
                            const std::vector<uint32_t>& relevance_ranks = {})
{
    const size_t label_count = group_posting_rows.size();
    require(graph_rows.size() == label_count && lowfreq_rows.size() == label_count,
            "fixture metadata lengths differ");

    GraphRuntime runtime;
    runtime.vector_scalar_type = DenseVectorScalarType::Uint8;
    runtime.original_query_count = 1;
    runtime.original_query_vectors.rows = 1;
    runtime.original_query_vectors.cols = 2;
    runtime.original_query_vectors.data = {7, 11};
    runtime.original_query_labels_matrix.rows = 1;
    runtime.original_query_labels_matrix.cols = static_cast<uint32_t>(label_count);
    runtime.original_query_labels_matrix.indptr = {0, labels.size()};
    runtime.original_query_labels_matrix.indices = labels;
    runtime.original_query_labels_matrix.row_labels = {labels};
    runtime.original_query_labels_matrix.source_tiers = source_tiers;
    runtime.original_query_labels_matrix.relevance_ranks = relevance_ranks;
    require((source_tiers.empty() && relevance_ranks.empty()) ||
                (source_tiers.size() == labels.size() &&
                 relevance_ranks.size() == labels.size()),
            "fixture occurrence metadata lengths differ");
    runtime.original_query_label_ops = {op};
    runtime.original_query_ranges.rows = 1;
    runtime.original_query_ranges.range_count = 0;

    runtime.graph.cols = graph_degree;
    runtime.group_layout.label_to_group.resize(label_count);
    runtime.group_layout.groups.resize(label_count);
    runtime.metadata.label_count = static_cast<uint32_t>(label_count);
    runtime.metadata.original_label_count = static_cast<uint32_t>(label_count);
    runtime.metadata.group_posting_size = group_posting_rows;
    runtime.metadata.cat_freq = group_posting_rows;
    runtime.metadata.label_size = graph_rows;
    runtime.metadata.label_offset.assign(label_count, 0);
    runtime.metadata.label_filter_bit.assign(label_count, 1);
    runtime.metadata.original_label_cardinality = original_label_cardinality.empty()
        ? group_posting_rows
        : original_label_cardinality;
    require(runtime.metadata.original_label_cardinality.size() == label_count,
            "fixture original-label cardinality length differs");
    runtime.lowfreq_index.group_count = static_cast<uint32_t>(label_count);
    runtime.lowfreq_index.label_size = lowfreq_rows;
    runtime.seed_local_id_by_label.assign(label_count, UINT32_MAX);
    runtime.seed_by_original_label.assign(label_count, UINT32_MAX);
    for (size_t label = 0; label < label_count; ++label) {
        runtime.group_layout.label_to_group[label] = static_cast<uint32_t>(label);
        runtime.group_layout.groups[label] = {static_cast<uint32_t>(label)};
        if (graph_rows[label] > 0) {
            runtime.seed_local_id_by_label[label] = 0;
            runtime.seed_by_original_label[label] = 0;
        }
    }
    runtime.and_dual_route_posting_ratio_threshold = 1.0;
    return runtime;
}

GraphRuntime route(GraphRuntime runtime)
{
    validate_and_prepare_queries(runtime, 2, "greedy_min_posting");
    validate_graph_runtime_query_metadata(runtime, "or_route_planner_test");
    return runtime;
}

struct RouteTuple {
    uint32_t original_query = UINT32_MAX;
    uint32_t filter_label = UINT32_MAX;
    uint32_t group = UINT32_MAX;
    uint64_t filter_bit = 0;
    uint8_t is_lowfreq = 0;
    uint32_t start_local_id = UINT32_MAX;
    std::vector<uint32_t> residual_labels;
    std::vector<uint8_t> range_modes;
    std::vector<float> range_lower_bounds;
    std::vector<float> range_upper_bounds;

    auto fields() const
    {
        return std::tie(original_query,
                        filter_label,
                        group,
                        filter_bit,
                        is_lowfreq,
                        start_local_id,
                        residual_labels,
                        range_modes,
                        range_lower_bounds,
                        range_upper_bounds);
    }

    bool operator==(const RouteTuple& other) const { return fields() == other.fields(); }
    bool operator<(const RouteTuple& other) const { return fields() < other.fields(); }
};

std::vector<RouteTuple> route_tuples(const GraphRuntime& runtime)
{
    std::vector<RouteTuple> routes;
    for (size_t idx = 0; idx < runtime.query_filter_labels.size(); ++idx) {
        RouteTuple route;
        route.original_query = runtime.route_to_original_query[idx];
        route.filter_label = runtime.query_filter_labels[idx];
        route.group = runtime.query_group_ids[idx];
        route.filter_bit = runtime.query_filter_bits[idx];
        route.is_lowfreq = runtime.query_is_lowfreq[idx];
        route.start_local_id = runtime.start_local_ids[idx];
        route.residual_labels = runtime.query_labels_matrix.row_labels[idx];
        for (uint32_t range_idx = 0; range_idx < runtime.query_ranges.range_count; ++range_idx) {
            const size_t flat = runtime.query_ranges.flat_index(idx, range_idx);
            route.range_modes.push_back(runtime.query_ranges.modes[flat]);
            route.range_lower_bounds.push_back(runtime.query_ranges.lower_bounds[flat]);
            route.range_upper_bounds.push_back(runtime.query_ranges.upper_bounds[flat]);
        }
        routes.push_back(std::move(route));
    }
    return routes;
}

std::vector<RouteTuple> sorted_route_tuples(const GraphRuntime& runtime)
{
    std::vector<RouteTuple> routes = route_tuples(runtime);
    std::sort(routes.begin(), routes.end());
    return routes;
}

std::string temporary_config_path(const char* suffix)
{
    return "/tmp/flashanns_or_route_planner_test_" + std::to_string(getpid()) + suffix;
}

std::vector<nlohmann::json> read_jsonl(const std::string& path)
{
    std::ifstream input(path);
    require(static_cast<bool>(input), "unable to open JSONL fixture " + path);
    std::vector<nlohmann::json> rows;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            rows.push_back(nlohmann::json::parse(line));
        }
    }
    return rows;
}

nlohmann::json hybrid_label_atom(const std::string& family,
                                 const std::string& subtype,
                                 uint32_t token_id)
{
    return {{"family", family},
            {"subtype", subtype},
            {"predicates", {{{"field", "caption_token"},
                              {"op", family == "categorical" ? "eq" : "match"},
                              {"match_mode", subtype == "exact" ? "exact" : ""},
                              {"token_id", token_id},
                              {"token_text", "token" + std::to_string(token_id)}}}}};
}

nlohmann::json hybrid_fuzzy_atom()
{
    return {{"family", "string"},
            {"subtype", "fuzzy"},
            {"predicates", {{{"field", "caption_token"},
                              {"op", "match"},
                              {"match_mode", "contains"},
                              {"pattern", "amp"},
                              {"source_token_id", 9},
                              {"source_token_text", "sample"}}}}};
}

nlohmann::json hybrid_range_atom(const std::string& subtype = "width")
{
    nlohmann::json predicates = nlohmann::json::array();
    predicates.push_back({{"column", 0},
                          {"field", "original_width"},
                          {"op", "between"},
                          {"lo", 10.0},
                          {"hi", 20.0},
                          {"inclusive_min", true},
                          {"inclusive_max", true}});
    if (subtype == "width_height") {
        predicates.push_back({{"column", 1},
                              {"field", "original_height"},
                              {"op", "gt"},
                              {"lo", 5.0},
                              {"inclusive_min", false}});
    }
    return {{"family", "range"}, {"subtype", subtype}, {"predicates", predicates}};
}

std::string hybrid_manifest_row(const nlohmann::json& left,
                                const nlohmann::json& right,
                                const std::string& op)
{
    const nlohmann::json predicate = {{"op", op}, {"children", {left, right}}};
    nlohmann::json record = {{"family", "hybrid"},
                             {"query_id", "hybrid-test"},
                             {"hybrid_top_logic", op},
                             {"predicate_logic", op == "and" ? "AND" : "OR"},
                             {"predicate", predicate},
                             {"predicates", {predicate}}};
    return record.dump();
}

struct ExpectedHybridClause {
    std::vector<uint32_t> labels;
    uint32_t branch = 0;
    bool range = false;
};

void require_hybrid_clauses(const HybridManifestParseTestResult& parsed,
                            const std::vector<ExpectedHybridClause>& expected,
                            const std::string& context)
{
    require(parsed.clauses.rows == 1 && parsed.clauses.query_indptr ==
                std::vector<uint64_t>({0, expected.size()}),
            context + " query-to-clause CSR mismatch");
    require(parsed.clauses.clause_count() == expected.size(),
            context + " clause count mismatch");
    for (size_t clause = 0; clause < expected.size(); ++clause) {
        const uint64_t begin = parsed.clauses.label_indptr[clause];
        const uint64_t end = parsed.clauses.label_indptr[clause + 1];
        const std::vector<uint32_t> labels(parsed.clauses.labels.begin() + begin,
                                           parsed.clauses.labels.begin() + end);
        require(labels == expected[clause].labels,
                context + " clause labels mismatch at " + std::to_string(clause));
        require(parsed.clauses.top_branch_ids[clause] == expected[clause].branch,
                context + " branch id mismatch at " + std::to_string(clause));
        require((parsed.clauses.uses_original_range[clause] != 0) == expected[clause].range,
                context + " range flag mismatch at " + std::to_string(clause));
    }
}

void test_hybrid_manifest_ten_subtype_dnf()
{
    const auto categorical = hybrid_label_atom("categorical", "single", 0);
    const auto exact = hybrid_label_atom("string", "exact", 1);
    const auto fuzzy = hybrid_fuzzy_atom();
    const auto range = hybrid_range_atom("width_height");
    const std::vector<uint32_t> fuzzy_labels{2, 3};
    const std::vector<uint8_t> fuzzy_tiers{
        tier(QueryLabelSourceTier::Direct), tier(QueryLabelSourceTier::Direct)};
    const std::vector<uint32_t> fuzzy_ranks{0, 0};
    const auto parse = [&](const nlohmann::json& left,
                           const nlohmann::json& right,
                           const std::string& op) {
        const bool has_fuzzy = left.value("subtype", "") == "fuzzy" ||
                               right.value("subtype", "") == "fuzzy";
        return evaluate_hybrid_query_manifest_row_for_test(
            hybrid_manifest_row(left, right, op),
            has_fuzzy ? fuzzy_labels : std::vector<uint32_t>{},
            has_fuzzy ? fuzzy_tiers : std::vector<uint8_t>{},
            has_fuzzy ? fuzzy_ranks : std::vector<uint32_t>{},
            8);
    };

    require_hybrid_clauses(parse(categorical, range, "and"), {{{0}, 0, true}}, "C AND R");
    require_hybrid_clauses(parse(range, categorical, "and"), {{{0}, 0, true}}, "R AND C");
    require_hybrid_clauses(parse(categorical, range, "or"),
                           {{{0}, 0, false}, {{}, 1, true}}, "C OR R");
    require_hybrid_clauses(parse(range, categorical, "or"),
                           {{{}, 0, true}, {{0}, 1, false}}, "R OR C");
    require_hybrid_clauses(parse(categorical, exact, "and"), {{{0, 1}, 0, false}}, "C AND E");
    require_hybrid_clauses(parse(categorical, exact, "or"),
                           {{{0}, 0, false}, {{1}, 1, false}}, "C OR E");
    require_hybrid_clauses(parse(categorical, fuzzy, "and"),
                           {{{0, 2}, 0, false}, {{0, 3}, 0, false}}, "C AND F");
    require_hybrid_clauses(parse(categorical, fuzzy, "or"),
                           {{{0}, 0, false}, {{2}, 1, false}, {{3}, 1, false}}, "C OR F");
    require_hybrid_clauses(parse(range, exact, "and"), {{{1}, 0, true}}, "R AND E");
    require_hybrid_clauses(parse(exact, range, "and"), {{{1}, 0, true}}, "E AND R");
    require_hybrid_clauses(parse(range, exact, "or"),
                           {{{}, 0, true}, {{1}, 1, false}}, "R OR E");
    require_hybrid_clauses(parse(range, fuzzy, "and"),
                           {{{2}, 0, true}, {{3}, 0, true}}, "R AND F");
    require_hybrid_clauses(parse(fuzzy, range, "and"),
                           {{{2}, 0, true}, {{3}, 0, true}}, "F AND R");
    require_hybrid_clauses(parse(range, fuzzy, "or"),
                           {{{}, 0, true}, {{2}, 1, false}, {{3}, 1, false}}, "R OR F");

    nlohmann::json mismatched = nlohmann::json::parse(
        hybrid_manifest_row(categorical, range, "or"));
    mismatched["predicate_logic"] = "AND";
    require_throws_contains(
        [&] { (void)evaluate_hybrid_query_manifest_row_for_test(
                  mismatched.dump(), {}, {}, {}, 8); },
        "inconsistent", "mismatched hybrid boolean fields were accepted");
    require_throws_contains(
        [&] { (void)evaluate_hybrid_query_manifest_row_for_test(
                  hybrid_manifest_row(exact, fuzzy, "and"),
                  fuzzy_labels, fuzzy_tiers, fuzzy_ranks, 8); },
        "unsupported hybrid atom pair", "exact plus fuzzy pair was accepted");
    require_throws_contains(
        [&] { (void)evaluate_hybrid_query_manifest_row_for_test(
                  hybrid_manifest_row(categorical, fuzzy, "and"), {}, {}, {}, 8); },
        "fuzzy expansion is empty", "empty hybrid fuzzy expansion was accepted");
}

void configure_hybrid_clause_fixture(GraphRuntime& runtime,
                                     const std::vector<uint64_t>& query_indptr,
                                     const std::vector<uint64_t>& label_indptr,
                                     const std::vector<uint32_t>& labels,
                                     const std::vector<uint32_t>& branches,
                                     const std::vector<uint8_t>& ranges)
{
    runtime.original_query_clauses.rows = 1;
    runtime.original_query_clauses.query_indptr = query_indptr;
    runtime.original_query_clauses.branch_caps = {0};
    runtime.original_query_clauses.label_indptr = label_indptr;
    runtime.original_query_clauses.labels = labels;
    runtime.original_query_clauses.source_tiers.assign(
        labels.size(), tier(QueryLabelSourceTier::Exact));
    runtime.original_query_clauses.relevance_ranks.assign(labels.size(), 0);
    runtime.original_query_clauses.top_branch_ids = branches;
    runtime.original_query_clauses.uses_original_range = ranges;
    runtime.original_query_ranges.range_count = 1;
    runtime.original_query_ranges.modes = {3};
    runtime.original_query_ranges.lower_bounds = {10.0f};
    runtime.original_query_ranges.upper_bounds = {20.0f};
}

void test_hybrid_clause_local_route_materialization(const ScopedRouteCaps& caps)
{
    GraphRuntime disjunction = make_runtime(
        {0}, QueryLabelOp::Or, {10}, {10}, {0});
    configure_hybrid_clause_fixture(disjunction,
                                    {0, 2}, {0, 1, 1}, {0}, {0, 1}, {0, 1});
    disjunction.supported_query_route_providers =
        kQueryRouteProviderLabel | kQueryRouteProviderRangePosting;
    const GraphRuntime routed_or = route(disjunction);
    require(routed_or.route_providers ==
                std::vector<QueryRouteProvider>({QueryRouteProvider::Label,
                                                 QueryRouteProvider::RangePosting}),
            "C OR R did not preserve distinct physical providers");
    require(routed_or.route_execution_modes ==
                std::vector<QueryRouteExecution>({QueryRouteExecution::GraphTraversal,
                                                  QueryRouteExecution::GpuResidentScan}) &&
                routed_or.route_candidate_sources ==
                    std::vector<QueryCandidateSource>({QueryCandidateSource::None,
                                                       QueryCandidateSource::RangePosting}),
            "C OR R did not preserve resident range execution metadata");
    require(routed_or.route_to_clause == std::vector<uint32_t>({0, 1}) &&
                routed_or.route_top_branch_ids == std::vector<uint32_t>({0, 1}),
            "C OR R lost clause/branch identity");
    require(routed_or.query_ranges.modes == std::vector<uint8_t>({0, 3}),
            "C OR R leaked range residual into the label branch");
    require(routed_or.high_query_indices == std::vector<uint32_t>({0}) &&
                routed_or.low_query_indices.empty() &&
                routed_or.query_group_ids[1] == UINT32_MAX,
            "range posting placeholder entered a label execution queue");
    const GraphRouteSplitStats split = collect_graph_route_split_stats(routed_or, {3, 4});
    require(split.highfreq_route_count == 1 &&
                split.provider_route_count == 1 &&
                split.unroutable_route_count == 0,
            "range posting provider was reported as an unroutable label route");
    caps.set(1, 1);
    require_throws_contains(
        [&] { (void)route(disjunction); },
        "range-bearing DNF branches cannot be capped",
        "range-bearing hybrid OR silently accepted an incomparable branch cap");
    caps.clear();

    GraphRuntime conjunction = make_runtime(
        {0}, QueryLabelOp::And, {10}, {10}, {0});
    configure_hybrid_clause_fixture(conjunction,
                                    {0, 1}, {0, 1}, {0}, {0}, {1});
    conjunction.supported_query_route_providers =
        kQueryRouteProviderLabel | kQueryRouteProviderRangePosting;
    const GraphRuntime routed_and = route(conjunction);
    require(routed_and.route_providers ==
                std::vector<QueryRouteProvider>({QueryRouteProvider::Label,
                                                 QueryRouteProvider::RangePosting}) &&
                routed_and.query_ranges.modes == std::vector<uint8_t>({3, 3}),
            "C AND R did not expose label and range enumeration alternatives");
    require(routed_and.route_execution_modes ==
                std::vector<QueryRouteExecution>({QueryRouteExecution::GraphTraversal,
                                                  QueryRouteExecution::HostCandidateStream}) &&
                routed_and.route_candidate_sources ==
                    std::vector<QueryCandidateSource>({QueryCandidateSource::None,
                                                       QueryCandidateSource::RangePosting}),
            "C AND R did not mark the range alternative as a generic candidate stream");

    disjunction.supported_query_route_providers = kQueryRouteProviderLabel;
    require_throws_contains(
        [&] { (void)route(disjunction); },
        "unsupported route provider",
        "range-only OR branch ran without backend range-posting capability");
}

uint64_t fnv_mix_u64(uint64_t hash, uint64_t value)
{
    constexpr uint64_t prime = 1099511628211ULL;
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & 0xffULL;
        hash *= prime;
    }
    return hash;
}

uint64_t fixture_group_layout_checksum(const GroupLayout& layout)
{
    uint64_t hash = 1469598103934665603ULL;
    hash = fnv_mix_u64(hash, layout.groups.size());
    hash = fnv_mix_u64(hash, layout.label_to_group.size());
    for (const auto& group : layout.groups) {
        hash = fnv_mix_u64(hash, group.size());
        for (uint32_t label : group) {
            hash = fnv_mix_u64(hash, label);
        }
    }
    return hash;
}

uint64_t fixture_hash(uint64_t sample_id, uint64_t seed)
{
    uint64_t value = sample_id ^ seed;
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

std::string repeated_hex(unsigned char value)
{
    static constexpr char digits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (size_t idx = 0; idx < 32; ++idx) {
        out.push_back(digits[value >> 4]);
        out.push_back(digits[value & 0xf]);
    }
    return out;
}

struct ModelFixtureOptions {
    bool validated = true;
    bool add_unknown_field = false;
    bool omit_base_coverage_weight = false;
    uint64_t training_sample_count = 1000;
    uint64_t training_query_count = 100;
    uint64_t validation_sample_count = 250;
    uint64_t validation_query_count = 25;
    uint64_t calibration_run_count = 3;
    double lowfreq_per_row_us = 1.0;
    double highfreq_per_work_us = 1.0;
    double exact_prior = 1.0;
    double direct_prior = 1.0;
    double fallback_prior = 1.0;
    double fallback_rank_decay = 0.2;
    double base_coverage_weight = 1.0;
    double marginal_coverage_weight = 10.0;
    double cost_weight = 0.01;
    uint64_t selection_route_cap = 0;
    uint64_t selection_highfreq_cap = 0;
    uint64_t utility_search_candidates = 8;
    uint64_t heldout_gt_coverage = 900;
    uint64_t heldout_baseline_gt_coverage = 1000;
    double heldout_gt_coverage_ratio = 0.9;
    double heldout_predicted_latency_us = 90.0;
    double heldout_baseline_predicted_latency_us = 100.0;
    double heldout_predicted_latency_ratio = 0.9;
    double min_required_gt_coverage_ratio = 0.8;
    double max_allowed_latency_ratio = 1.05;
};

void write_model_fixture(const std::string& path,
                         const GraphRuntime& runtime,
                         uint32_t sketch_k,
                         uint64_t hash_seed,
                         const ModelFixtureOptions& options = {})
{
    const uint64_t num_points = runtime.base_vector_metadata.rows;
    nlohmann::json model = {
        {"schema", "flashanns.or_route_accuracy_cost_model"},
        {"version", 1},
        {"calibration_id", "synthetic-heldout-v1"},
        {"validated", options.validated},
        {"training_sample_count", options.training_sample_count},
        {"training_query_count", options.training_query_count},
        {"validation_sample_count", options.validation_sample_count},
        {"validation_query_count", options.validation_query_count},
        {"calibration_run_count", options.calibration_run_count},
        {"cost_model", {
            {"latency_unit", "microseconds"},
            {"lowfreq_intercept_us", 1.0},
            {"lowfreq_per_row_us", options.lowfreq_per_row_us},
            {"highfreq_intercept_us", 2.0},
            {"highfreq_per_work_us", options.highfreq_per_work_us},
            {"cost_floor_us", 0.1},
        }},
        {"quality_model", {
            {"exact_prior", options.exact_prior},
            {"direct_prior", options.direct_prior},
            {"fallback_prior", options.fallback_prior},
            {"fallback_rank_decay", options.fallback_rank_decay},
            {"base_coverage_weight", options.base_coverage_weight},
            {"marginal_coverage_weight", options.marginal_coverage_weight},
            {"marginal_coverage_exponent", 1.0},
        }},
        {"utility", {
            {"formula", "quality_weight*quality-cost_weight*(max(predicted_latency_us,cost_floor_us)/latency_scale_us)^cost_exponent"},
            {"quality_weight", 1.0},
            {"cost_weight", options.cost_weight},
            {"latency_scale_us", 1.0},
            {"cost_exponent", 1.0},
        }},
        {"fit_diagnostics", {
            {"heldout_query_fraction", 0.2},
            {"cost_r2", 0.99},
            {"quality_r2", 0.9},
            {"cost_mae_us", 0.1},
            {"quality_mae", 0.1},
            {"split_seed", 17},
            {"lowfreq_condition_number", 10.0},
            {"highfreq_condition_number", 10.0},
            {"quality_condition_number", 10.0},
            {"selection_route_cap", options.selection_route_cap},
            {"selection_highfreq_cap", options.selection_highfreq_cap},
            {"utility_search_candidates", options.utility_search_candidates},
            {"heldout_gt_coverage", options.heldout_gt_coverage},
            {"heldout_baseline_gt_coverage", options.heldout_baseline_gt_coverage},
            {"heldout_gt_coverage_ratio", options.heldout_gt_coverage_ratio},
            {"heldout_predicted_latency_us", options.heldout_predicted_latency_us},
            {"heldout_baseline_predicted_latency_us",
             options.heldout_baseline_predicted_latency_us},
            {"heldout_predicted_latency_ratio", options.heldout_predicted_latency_ratio},
            {"min_required_gt_coverage_ratio", options.min_required_gt_coverage_ratio},
            {"max_allowed_latency_ratio", options.max_allowed_latency_ratio},
            {"validation_errors", nlohmann::json::array()},
        }},
        {"sketch", {
            {"sidecar_schema", "flashanns.or_route_kmv_sidecar"},
            {"sidecar_version", 1},
            {"token_count", runtime.metadata.original_label_count},
            {"num_points", num_points},
            {"max_row_id_exclusive", num_points},
            {"sketch_k", sketch_k},
            {"hash_seed", hash_seed},
            {"group_count", runtime.group_layout.groups.size()},
            {"group_layout_checksum", fixture_group_layout_checksum(runtime.group_layout)},
            {"payload_sha256", repeated_hex(0x22)},
            {"source_posting_sha256", repeated_hex(0x11)},
        }},
        {"provenance", {
            {"tool", "or-route-planner-test"},
            {"created_utc", "2026-07-14T00:00:00Z"},
            {"trace_inputs", nlohmann::json::array({"synthetic.jsonl"})},
            {"source", "route_level_heldout_query_split"},
        }},
    };
    if (options.add_unknown_field) {
        model["unknown"] = 1;
    }
    if (options.omit_base_coverage_weight) {
        model["quality_model"].erase("base_coverage_weight");
    }
    std::ofstream output(path);
    output << model.dump(2) << '\n';
}

void configure_marginal_fixture(const GraphRuntime& runtime,
                                const std::string& model_path,
                                uint32_t sketch_k,
                                uint64_t hash_seed,
                                const std::string& sidecar_path = {})
{
    (void)runtime;
    OrRoutePlannerConfig config;
    config.order_mode = "marginal_accuracy_cost";
    config.accuracy_cost_model_path = model_path;
    config.sketch_sidecar_path = sidecar_path;
    config.sketch_k = sketch_k;
    config.sketch_hash_seed = hash_seed;
    config.allow_runtime_sketch_fallback = sidecar_path.empty();
    config.runtime_sketch_fallback_max_rows = 1000;
    config.verify_sketch_payload_checksum = false;
    configure_or_route_planner(config);
}

GraphRuntime make_overlap_runtime()
{
    GraphRuntime runtime = make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {5, 4, 3}, {0, 0, 0}, {5, 4, 3},
        4, {5, 4, 3});
    runtime.base_vector_metadata.rows = 8;
    runtime.lowfreq_index.label_offset = {0, 5, 9};
    runtime.lowfreq_index.sample_ids = {
        0, 1, 2, 3, 4,
        0, 1, 2, 3,
        5, 6, 7,
    };
    return runtime;
}

GraphRuntime make_mixed_trace_runtime()
{
    GraphRuntime runtime = make_runtime(
        {0}, QueryLabelOp::And, {8, 2, 4}, {8, 0, 4}, {0, 2, 0}, 4, {8, 2, 4});
    runtime.original_query_count = 2;
    runtime.original_query_vectors.rows = 2;
    runtime.original_query_vectors.data = {7, 11, 13, 17};
    runtime.original_query_labels_matrix.rows = 2;
    runtime.original_query_labels_matrix.indptr = {0, 1, 3};
    runtime.original_query_labels_matrix.indices = {0, 1, 2};
    runtime.original_query_labels_matrix.row_labels = {{0}, {1, 2}};
    runtime.original_query_labels_matrix.source_tiers = {
        tier(QueryLabelSourceTier::Exact),
        tier(QueryLabelSourceTier::Direct),
        tier(QueryLabelSourceTier::Fallback),
    };
    runtime.original_query_labels_matrix.relevance_ranks = {0, 0, 1};
    runtime.original_query_label_ops = {QueryLabelOp::And, QueryLabelOp::Or};
    runtime.original_query_ranges.rows = 2;
    return runtime;
}

void write_sketch_sidecar_fixture(const std::string& path,
                                  const GraphRuntime& runtime,
                                  const std::vector<std::vector<uint32_t>>& samples,
                                  uint32_t sketch_k,
                                  uint64_t hash_seed,
                                  uint64_t header_num_points_override = 0,
                                  uint64_t header_layout_checksum_override = 0,
                                  bool truncate_file = false,
                                  bool corrupt_active_offsets = false)
{
    require(samples.size() == runtime.metadata.original_label_count,
            "sidecar fixture sample count differs from labels");
    std::vector<uint64_t> offsets(samples.size() + 1, 0);
    std::vector<uint64_t> hashes;
    uint64_t posting_count = 0;
    for (size_t label = 0; label < samples.size(); ++label) {
        std::vector<uint64_t> label_hashes;
        for (uint32_t sample_id : samples[label]) {
            label_hashes.push_back(fixture_hash(sample_id, hash_seed));
        }
        std::sort(label_hashes.begin(), label_hashes.end());
        if (label_hashes.size() > sketch_k) {
            label_hashes.resize(sketch_k);
        }
        hashes.insert(hashes.end(), label_hashes.begin(), label_hashes.end());
        offsets[label + 1] = hashes.size();
        posting_count += samples[label].size();
    }
    if (corrupt_active_offsets && offsets.size() > 2) {
        offsets[1] = offsets[2] + 1;
    }

    OrRouteSketchSidecarHeader header{};
    const char magic[16] = {'F','A','N','N','S','K','M','V','S','K','E','T','C','H','\0','\0'};
    std::copy(std::begin(magic), std::end(magic), std::begin(header.magic));
    header.version = 1;
    header.header_bytes = sizeof(header);
    header.endian = 0x01020304;
    header.flags = 1;
    header.token_count = samples.size();
    header.num_points = header_num_points_override == 0
        ? runtime.base_vector_metadata.rows
        : header_num_points_override;
    header.max_row_id_exclusive = header.num_points;
    header.sketch_k = sketch_k;
    header.hash_seed = hash_seed;
    header.hash_count = hashes.size();
    header.group_count = runtime.group_layout.groups.size();
    header.group_layout_checksum = header_layout_checksum_override == 0
        ? fixture_group_layout_checksum(runtime.group_layout)
        : header_layout_checksum_override;
    header.source_posting_version = 1;
    header.source_posting_token_count = samples.size();
    header.source_posting_count = posting_count;
    header.source_posting_max_row_id_exclusive = header.num_points;
    header.source_posting_file_bytes = 64 + (samples.size() + 1) * 8 + posting_count * 4;
    header.checksum_kind = 1;
    std::fill(std::begin(header.source_posting_sha256),
              std::end(header.source_posting_sha256), 0x11);
    std::fill(std::begin(header.payload_sha256),
              std::end(header.payload_sha256), 0x22);
    std::memcpy(header.provenance, "tokpost_v1", 10);

    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(offsets.data()),
                 static_cast<std::streamsize>(offsets.size() * sizeof(uint64_t)));
    output.write(reinterpret_cast<const char*>(hashes.data()),
                 static_cast<std::streamsize>(hashes.size() * sizeof(uint64_t)));
    output.close();
    if (truncate_file) {
        require(truncate(path.c_str(), static_cast<off_t>(
                    sizeof(header) + offsets.size() * sizeof(uint64_t))) == 0,
                "unable to truncate sidecar fixture");
    }
}

void test_config_loading()
{
    const std::string root_path = temporary_config_path("_root.json");
    const std::string nested_path = temporary_config_path("_nested.json");
    const std::string default_path = temporary_config_path("_default.json");
    {
        std::ofstream output(root_path);
        output << R"({"or_route_order_mode":"cost_aware","query_manifest_fuzzy_cap_policy":"coverage_cost","or_route_sketch_sidecar_path":"/root/sketch.bin","or_route_accuracy_cost_model_path":"/root/model.json","or_route_calibration_trace_path":"/root/trace.jsonl","or_route_calibration_run_id":"root-run","or_route_sketch_k":32,"or_route_sketch_hash_seed":123,"or_route_allow_runtime_sketch_fallback":true,"or_route_runtime_sketch_fallback_max_rows":99,"or_route_verify_sketch_payload_checksum":true})";
    }
    {
        std::ofstream output(nested_path);
        output << R"({"runtime":{"or_route_order_mode":"marginal_accuracy_cost","query_manifest_fuzzy_cap_policy":"relevance_coverage_cost","or_route_sketch_sidecar_path":"/nested/sketch.bin","or_route_accuracy_cost_model_path":"/nested/model.json","or_route_calibration_trace_path":"/nested/trace.jsonl","or_route_calibration_run_id":"nested-run","or_route_sketch_k":64,"or_route_sketch_hash_seed":456,"or_route_allow_runtime_sketch_fallback":false,"or_route_runtime_sketch_fallback_max_rows":1000,"or_route_verify_sketch_payload_checksum":false}})";
    }
    {
        std::ofstream output(default_path);
        output << "{}";
    }
    require(go_fann::load_config_file(root_path).or_route_order_mode == "cost_aware",
            "root planner config was not loaded");
    require(go_fann::load_config_file(nested_path).or_route_order_mode ==
                "marginal_accuracy_cost",
            "nested planner config was not loaded");
    const auto root = go_fann::load_config_file(root_path);
    const auto nested = go_fann::load_config_file(nested_path);
    require(root.query_manifest_fuzzy_cap_policy == "coverage_cost" &&
                root.or_route_sketch_sidecar_path == "/root/sketch.bin" &&
                root.or_route_accuracy_cost_model_path == "/root/model.json" &&
                root.or_route_calibration_trace_path == "/root/trace.jsonl" &&
                root.or_route_calibration_run_id == "root-run" &&
                root.or_route_sketch_k == 32 && root.or_route_sketch_hash_seed == 123 &&
                root.or_route_allow_runtime_sketch_fallback == true &&
                root.or_route_runtime_sketch_fallback_max_rows == 99 &&
                root.or_route_verify_sketch_payload_checksum == true,
            "root marginal planner fields were not loaded");
    require(nested.query_manifest_fuzzy_cap_policy == "relevance_coverage_cost" &&
                nested.or_route_sketch_sidecar_path == "/nested/sketch.bin" &&
                nested.or_route_accuracy_cost_model_path == "/nested/model.json" &&
                nested.or_route_calibration_trace_path == "/nested/trace.jsonl" &&
                nested.or_route_calibration_run_id == "nested-run" &&
                nested.or_route_sketch_k == 64 && nested.or_route_sketch_hash_seed == 456 &&
                nested.or_route_allow_runtime_sketch_fallback == false &&
                nested.or_route_runtime_sketch_fallback_max_rows == 1000 &&
                nested.or_route_verify_sketch_payload_checksum == false,
            "nested runtime marginal planner fields were not loaded");
    const auto defaults = go_fann::load_config_file(default_path);
    require(!defaults.or_route_order_mode.has_value(),
            "missing planner config did not preserve the application default");
    require(!defaults.query_manifest_fuzzy_cap_policy.has_value(),
            "missing fuzzy cap policy did not preserve the application default");
    require(defaults.or_route_verify_sketch_payload_checksum.value_or(false),
            "empty config did not select checksum verification by default");
    require(OrRoutePlannerConfig{}.verify_sketch_payload_checksum,
            "planner checksum verification default is not safe");
    require(is_valid_or_route_order_mode(""),
            "empty planner mode does not select the legacy default");
    require(is_valid_or_route_order_mode("coverage_cost"),
            "coverage_cost planner mode failed validation");
    require(is_valid_or_route_order_mode("relevance_coverage_cost"),
            "relevance_coverage_cost planner mode failed validation");
    require(is_valid_or_route_order_mode("marginal_accuracy_cost"),
            "marginal_accuracy_cost planner mode failed validation");
    require(!is_valid_or_route_order_mode("unknown"),
            "unknown planner mode passed validation");
    require(is_valid_query_manifest_fuzzy_cap_policy("legacy") &&
                is_valid_query_manifest_fuzzy_cap_policy("cost-aware") &&
                is_valid_query_manifest_fuzzy_cap_policy("coverage_cost") &&
                is_valid_query_manifest_fuzzy_cap_policy(
                    "relevance_coverage_cost") &&
                !is_valid_query_manifest_fuzzy_cap_policy("manifest") &&
                !is_valid_query_manifest_fuzzy_cap_policy(
                    "marginal_accuracy_cost"),
            "fuzzy cap policy validation accepted the wrong policy set");
    std::remove(root_path.c_str());
    std::remove(nested_path.c_str());
    std::remove(default_path.c_str());
}

void test_default_and_route_set_preservation()
{
    const GraphRuntime fixture = make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {8, 2, 99}, {8, 2, 0}, {0, 0, 1});

    const GraphRuntime legacy = route(fixture);
    require(legacy.query_filter_labels == std::vector<uint32_t>({0, 1, 2}),
            "default planner did not preserve legacy route order");
    require(legacy.or_route_planner_profile.mode == OrRouteOrderMode::Legacy,
            "default planner mode is not legacy");
    require(legacy.or_route_planner_profile.reordered_routes == 0,
            "legacy planner reported reordered routes");

    configure_or_route_planner("cost_aware");
    const GraphRuntime cost_aware = route(fixture);
    require(cost_aware.query_filter_labels == std::vector<uint32_t>({2, 1, 0}),
            "cost-aware mixed-tier order is incorrect");
    require(sorted_route_tuples(cost_aware) == sorted_route_tuples(legacy),
            "uncapped cost-aware planner changed route metadata or route multiset");
    require(cost_aware.or_route_planner_profile.original_routes == 3 &&
                cost_aware.or_route_planner_profile.emitted_routes == 3,
            "uncapped planner changed route count");
    require(cost_aware.or_route_planner_profile.reordered_queries == 1 &&
                cost_aware.or_route_planner_profile.reordered_routes == 2,
            "reorder counters are incorrect");
}

void test_lowfreq_exact_cost_ordering()
{
    configure_or_route_planner("cost_aware");
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {1, 100, 2}, {0, 0, 0}, {3, 9, 1}));
    require(runtime.query_filter_labels == std::vector<uint32_t>({2, 0, 1}),
            "low-frequency order did not use exact lowfreq posting lengths");
    require(runtime.or_route_planner_profile.original_exact_lowfreq_rows == 13 &&
                runtime.or_route_planner_profile.emitted_exact_lowfreq_rows == 13,
            "low-frequency row profile is incorrect");
}

void test_duplicate_group_filter_and_residual_metadata_preservation()
{
    GraphRuntime fixture = make_runtime(
        {0, 1, 0, 2}, QueryLabelOp::Or, {50, 0, 7}, {10, 0, 0}, {0, 0, 3});
    fixture.group_layout.label_to_group = {0, 0, 2};
    fixture.group_layout.groups = {{0, 1}, {}, {2}};
    fixture.metadata.label_filter_bit = {1, 2, 4};
    fixture.original_query_ranges.range_count = 1;
    fixture.original_query_ranges.modes = {1};
    fixture.original_query_ranges.lower_bounds = {0.25f};
    fixture.original_query_ranges.upper_bounds = {0.75f};

    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(fixture);
    configure_or_route_planner("cost_aware");
    const GraphRuntime cost_aware = route(fixture);

    require(legacy.query_filter_labels == std::vector<uint32_t>({0, 1, 0, 2}),
            "legacy planner changed duplicate/shared-group route order");
    require(cost_aware.query_filter_labels == std::vector<uint32_t>({2, 0, 1, 0}),
            "cost-aware duplicate/shared-group order is incorrect");
    require(sorted_route_tuples(cost_aware) == sorted_route_tuples(legacy),
            "cost-aware planner changed duplicate, group, filter, or residual metadata");
}

void test_highfreq_proxy_ordering_and_ties()
{
    configure_or_route_planner("cost_aware");
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {1, 100, 2}, {5, 1, 3}, {0, 0, 0}));
    require(runtime.query_filter_labels == std::vector<uint32_t>({1, 2, 0}),
            "high-frequency order did not use graph rows times degree");
    require(runtime.or_route_planner_profile.original_highfreq_cost_proxy == 36 &&
                runtime.or_route_planner_profile.emitted_highfreq_cost_proxy == 36,
            "high-frequency proxy profile is incorrect");

    const GraphRuntime tied = route(make_runtime(
        {2, 0, 1}, QueryLabelOp::Or, {9, 8, 7}, {2, 2, 2}, {0, 0, 0}));
    require(tied.query_filter_labels == std::vector<uint32_t>({2, 0, 1}),
            "equal-cost routes did not preserve original position");
}

void test_coverage_retention_then_cost_scheduling(const ScopedRouteCaps& caps)
{
    configure_or_route_planner("coverage_cost");
    caps.set(2, 1);
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2},
        QueryLabelOp::Or,
        {1, 10, 100},
        {0, 0, 100},
        {1, 10, 0}));
    caps.clear();

    require(runtime.query_filter_labels == std::vector<uint32_t>({1, 2}),
            "coverage retention did not keep the largest routes before cost scheduling");
    const OrRoutePlannerProfile& profile = runtime.or_route_planner_profile;
    require(profile.original_exact_label_cardinality_sum == 111 &&
                profile.emitted_exact_label_cardinality_sum == 110,
            "cardinality-sum profile did not distinguish original and emitted routes");
    require(profile.original_highfreq_routes == 1 && profile.emitted_highfreq_routes == 1,
            "larger high-frequency coverage was not retained");
    require(profile.capped_routes == 1 && profile.total_cap_capped_routes == 1,
            "coverage retention cap accounting is incorrect");
}

void test_relevance_retention_then_cost_scheduling(const ScopedRouteCaps& caps)
{
    configure_or_route_planner("relevance_coverage_cost");
    caps.set(3, 3);
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2, 3},
        QueryLabelOp::Or,
        {10, 20, 1000, 30},
        {0, 0, 0, 0},
        {10, 20, 1, 5},
        4,
        {},
        {tier(QueryLabelSourceTier::Fallback),
         tier(QueryLabelSourceTier::Direct),
         tier(QueryLabelSourceTier::Fallback),
         tier(QueryLabelSourceTier::Direct)},
        {1, 0, 2, 0}));
    caps.clear();

    require(runtime.query_filter_labels == std::vector<uint32_t>({3, 0, 1}),
            "relevance retention or physical-cost scheduling is incorrect");
    const OrRoutePlannerProfile& profile = runtime.or_route_planner_profile;
    require(profile.original_direct_source_routes == 2 &&
                profile.emitted_direct_source_routes == 2 &&
                profile.original_fallback_source_routes == 2 &&
                profile.emitted_fallback_source_routes == 1,
            "direct/fallback route diagnostics are incorrect");
    require(profile.retained_relevance_rank_sum == 1 &&
                profile.retained_relevance_rank_max == 1,
            "retained relevance-rank diagnostics are incorrect");
    require(profile.emitted_exact_label_cardinality_sum == 60,
            "relevance retention kept the wrong route set");
}

void test_equal_relevance_uses_cardinality_before_cost(const ScopedRouteCaps& caps)
{
    configure_or_route_planner("relevance_coverage_cost");
    caps.set(2, 2);
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2},
        QueryLabelOp::Or,
        {10, 100, 50},
        {0, 0, 0},
        {1, 20, 5},
        4,
        {},
        {tier(QueryLabelSourceTier::Fallback),
         tier(QueryLabelSourceTier::Fallback),
         tier(QueryLabelSourceTier::Fallback)},
        {1, 1, 1}));
    caps.clear();

    require(runtime.query_filter_labels == std::vector<uint32_t>({2, 1}),
            "equal-relevance retention did not use cardinality before cost scheduling");
}

void test_categorical_or_defaults_to_exact_relevance(const ScopedRouteCaps& caps)
{
    configure_or_route_planner("relevance_coverage_cost");
    caps.set(1, 1);
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {10, 100, 50}, {0, 0, 0}, {1, 20, 5}));
    caps.clear();

    require(runtime.query_filter_labels == std::vector<uint32_t>({1}),
            "categorical OR without manifest relevance did not default to exact coverage");
    require(runtime.or_route_planner_profile.original_exact_source_routes == 3 &&
                runtime.or_route_planner_profile.emitted_exact_source_routes == 1,
            "categorical exact-source diagnostics are incorrect");
}

void test_manifest_categorical_cap_uses_coverage(const ScopedRouteCaps& caps)
{
    configure_or_route_planner("legacy");
    caps.clear();
    GraphRuntime fixture = make_runtime(
        {0, 1, 2},
        QueryLabelOp::Or,
        {10, 100, 50},
        {10, 100, 50},
        {0, 0, 0},
        4,
        {10, 100, 50});
    fixture.original_query_clauses.rows = 1;
    fixture.original_query_clauses.query_indptr = {0, 3};
    fixture.original_query_clauses.branch_caps = {2};
    fixture.original_query_clauses.label_indptr = {0, 1, 2, 3};
    fixture.original_query_clauses.labels = {0, 1, 2};
    fixture.original_query_clauses.source_tiers.assign(
        3, tier(QueryLabelSourceTier::Exact));
    fixture.original_query_clauses.relevance_ranks.assign(3, 0);
    fixture.original_query_clauses.top_branch_ids = {0, 1, 2};
    fixture.original_query_clauses.uses_original_range = {0, 0, 0};

    const GraphRuntime runtime = route(std::move(fixture));
    require(runtime.query_filter_labels == std::vector<uint32_t>({1, 2}),
            "manifest categorical OR cap did not retain the coverage-selected branch set");
    require(runtime.route_to_clause == std::vector<uint32_t>({1, 2}) &&
                runtime.or_route_planner_profile.exact_cardinality_used,
            "manifest categorical OR did not preserve logical clause order after selection");
    require(runtime.or_route_planner_profile.capped_routes == 1 &&
                runtime.or_route_planner_profile.total_cap_capped_routes == 1 &&
                runtime.or_route_planner_profile.manifest_cap_queries == 1 &&
                runtime.or_route_planner_profile.manifest_cap_max == 2,
            "manifest categorical OR cap accounting is incorrect");
}

void test_uncapped_relevance_mode_preserves_multiset()
{
    const GraphRuntime fixture = make_runtime(
        {0, 1, 2},
        QueryLabelOp::Or,
        {8, 2, 99},
        {8, 2, 0},
        {0, 0, 1},
        4,
        {},
        {tier(QueryLabelSourceTier::Fallback),
         tier(QueryLabelSourceTier::Direct),
         tier(QueryLabelSourceTier::Fallback)},
        {2, 0, 1});
    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(fixture);
    configure_or_route_planner("relevance_coverage_cost");
    const GraphRuntime relevance = route(fixture);

    require(sorted_route_tuples(relevance) == sorted_route_tuples(legacy),
            "uncapped relevance mode changed the complete route multiset");
}

GraphRuntime make_fuzzy_manifest_cap_fixture(bool include_exact_branch = false)
{
    GraphRuntime fixture = make_runtime(
        {2, 0, 1}, QueryLabelOp::Or, {10, 100, 50}, {1, 20, 10}, {0, 0, 0},
        4, {10, 100, 50});
    fixture.original_query_clauses.rows = 1;
    fixture.original_query_clauses.query_indptr = include_exact_branch
        ? std::vector<uint64_t>({0, 4})
        : std::vector<uint64_t>({0, 3});
    fixture.original_query_clauses.branch_caps = {1};
    fixture.original_query_clauses.branch_cap_is_fuzzy = {1};
    fixture.original_query_clauses.label_indptr = include_exact_branch
        ? std::vector<uint64_t>({0, 1, 2, 3, 4})
        : std::vector<uint64_t>({0, 1, 2, 3});
    fixture.original_query_clauses.labels = include_exact_branch
        ? std::vector<uint32_t>({0, 2, 0, 1})
        : std::vector<uint32_t>({2, 0, 1});
    fixture.original_query_clauses.source_tiers.assign(
        fixture.original_query_clauses.labels.size(), tier(QueryLabelSourceTier::Direct));
    if (include_exact_branch) {
        fixture.original_query_clauses.source_tiers.front() = tier(QueryLabelSourceTier::Exact);
    }
    fixture.original_query_clauses.relevance_ranks.assign(
        fixture.original_query_clauses.labels.size(), 0);
    fixture.original_query_clauses.top_branch_ids.resize(
        fixture.original_query_clauses.labels.size());
    for (uint32_t idx = 0; idx < fixture.original_query_clauses.top_branch_ids.size(); ++idx) {
        fixture.original_query_clauses.top_branch_ids[idx] = idx;
    }
    fixture.original_query_clauses.uses_original_range.assign(
        fixture.original_query_clauses.labels.size(), 0);
    return fixture;
}

void test_fuzzy_cap_policy_selects_label_identity()
{
    const auto configure_fuzzy_policy = [](const std::string& policy) {
        OrRoutePlannerConfig config;
        config.fuzzy_cap_policy = policy;
        configure_or_route_planner(config);
    };
    configure_fuzzy_policy("legacy");
    const GraphRuntime legacy = route(make_fuzzy_manifest_cap_fixture());
    configure_fuzzy_policy("cost_aware");
    const GraphRuntime cost = route(make_fuzzy_manifest_cap_fixture());
    configure_fuzzy_policy("coverage_cost");
    const GraphRuntime coverage = route(make_fuzzy_manifest_cap_fixture());

    require(legacy.query_filter_labels == std::vector<uint32_t>({2}),
            "legacy fuzzy cap did not retain the first logical alternative");
    require(cost.query_filter_labels == std::vector<uint32_t>({0}),
            "cost-aware fuzzy cap did not select the cheapest physical alternative");
    require(coverage.query_filter_labels == std::vector<uint32_t>({1}),
            "coverage fuzzy cap did not select the largest exact-cardinality alternative");

    configure_fuzzy_policy("coverage_cost");
    const GraphRuntime with_exact = route(make_fuzzy_manifest_cap_fixture(true));
    require(with_exact.query_filter_labels == std::vector<uint32_t>({0, 1}),
            "an explicit exact branch competed with the fuzzy semantic-term budget");

    OrRoutePlannerConfig isolated;
    isolated.order_mode = "legacy";
    isolated.fuzzy_cap_policy = "cost_aware";
    configure_or_route_planner(isolated);
    const GraphRuntime isolated_fuzzy = route(make_fuzzy_manifest_cap_fixture());
    const GraphRuntime uncapped_categorical = route(make_runtime(
        {2, 0, 1}, QueryLabelOp::Or, {10, 100, 50}, {1, 20, 10}, {0, 0, 0}));
    require(isolated_fuzzy.query_filter_labels == std::vector<uint32_t>({0}),
            "isolated fuzzy cap policy did not select the cheapest fuzzy identity");
    require(uncapped_categorical.query_filter_labels ==
                std::vector<uint32_t>({2, 0, 1}),
            "fuzzy cap policy leaked into uncapped categorical OR ordering");

    GraphRuntime ranked = make_fuzzy_manifest_cap_fixture();
    ranked.original_query_labels_matrix.source_tiers.assign(
        3, tier(QueryLabelSourceTier::Direct));
    ranked.original_query_labels_matrix.relevance_ranks = {2, 0, 1};
    ranked.original_query_clauses.relevance_ranks = {2, 0, 1};
    configure_fuzzy_policy("relevance_coverage_cost");
    const GraphRuntime relevance = route(std::move(ranked));
    require(relevance.query_filter_labels == std::vector<uint32_t>({0}),
            "literal direct relevance rank did not select the closest fuzzy term");
}

void test_pre_dnf_fuzzy_cap()
{
    const char* old = std::getenv("GO_FANN_PRE_DNF_FUZZY_CAP");
    const std::optional<std::string> saved = old ? std::optional<std::string>(old) : std::nullopt;
    const auto fixture = [](bool conjunction, bool unary) {
        auto runtime = make_runtime({0, 1, 2}, QueryLabelOp::Or,
            {20, 100, 10}, {20, 100, 10}, {0, 0, 0}, 4, {20, 100, 10});
        auto& table = runtime.original_query_clauses;
        table.virtual_descriptors = true;
        table.query_indptr = {0}; table.descriptor_indptr = {0}; table.label_indptr = {0};
        table.atom_labels = {0, 1, 2, 2};
        table.atom_source_tiers = {1, 1, 1, 0}; table.atom_relevance_ranks = {0, 0, 0, 0};
        HybridClauseExpression expression;
        expression.active = true; expression.unary = unary; expression.conjunction = conjunction;
        expression.atoms[0] = {0, 3, false, true}; expression.atoms[1] = {3, 1, false, false};
        expression.branch_ids = {0, 1};
        table.expressions = {expression}; table.finish_clause_row(); table.rows = 1;
        table.branch_caps = {1}; table.branch_cap_is_fuzzy = {1};
        return runtime;
    };
    OrRoutePlannerConfig config; config.fuzzy_cap_policy = "coverage_cost";
    configure_or_route_planner(config);
    for (uint32_t cap : {1U, 2U, 12U}) {
        auto source = fixture(false, true); source.original_query_clauses.branch_caps = {cap};
        unsetenv("GO_FANN_PRE_DNF_FUZZY_CAP"); const auto baseline = route(source);
        setenv("GO_FANN_PRE_DNF_FUZZY_CAP", "1", 1); const auto early = route(source);
        require(route_tuples(baseline) == route_tuples(early) &&
            baseline.or_route_planner_profile.original_routes == early.or_route_planner_profile.original_routes &&
            baseline.or_route_planner_profile.capped_routes == early.or_route_planner_profile.capped_routes,
            "early unary cap changed routes or budget counts");
        require(early.original_query_clauses.atom_labels == source.original_query_clauses.atom_labels,
            "early cap altered the full predicate");
    }
    const auto and_source = fixture(true, false);
    unsetenv("GO_FANN_PRE_DNF_FUZZY_CAP"); const auto baseline = route(and_source);
    setenv("GO_FANN_PRE_DNF_FUZZY_CAP", "1", 1); const auto early = route(and_source);
    require(baseline.route_to_clause == std::vector<uint32_t>({0, 2}) &&
        early.route_to_clause == std::vector<uint32_t>({1, 2}),
        "early AND cap did not use fuzzy atom coverage or preserve exact duplicate");
    require(early.original_query_clauses.clause_count() == 3 &&
        early.original_query_clauses.atom_labels == and_source.original_query_clauses.atom_labels,
        "early AND cap truncated original predicate alternatives");
    const auto or_source = fixture(false, false); const auto early_or = route(or_source);
    require(early_or.route_to_clause == std::vector<uint32_t>({1, 3}),
        "early OR cap dropped its mandatory exact sibling");
    auto right_fuzzy = or_source;
    std::swap(right_fuzzy.original_query_clauses.expressions[0].atoms[0],
              right_fuzzy.original_query_clauses.expressions[0].atoms[1]);
    const auto early_right = route(right_fuzzy);
    require(early_right.route_to_clause == std::vector<uint32_t>({0, 2}),
        "early OR cap did not offset right-side fuzzy alternatives");
    if (saved) setenv("GO_FANN_PRE_DNF_FUZZY_CAP", saved->c_str(), 1);
    else unsetenv("GO_FANN_PRE_DNF_FUZZY_CAP");
}

void test_singleton_dnf_large_cap_and_lowfreq()
{
    std::vector<uint32_t> labels(24), branches(24);
    std::vector<uint64_t> offsets(25);
    for (uint32_t i = 0; i < 24; ++i) {
        labels[i] = branches[i] = i;
        offsets[i] = i;
    }
    offsets.back() = 24;
    std::vector<uint32_t> graph(24, 10), low(24, 0);
    graph[5] = 0;
    low[5] = 100;
    GraphRuntime fixture = make_runtime(
        labels, QueryLabelOp::Or, std::vector<uint32_t>(24, 100), graph, low);
    configure_hybrid_clause_fixture(fixture, {0, 24}, offsets, labels, branches,
                                    std::vector<uint8_t>(24, 0));
    fixture.original_query_clauses.branch_cap_is_fuzzy = {1};
    fixture.original_query_clauses.source_tiers.assign(24, tier(QueryLabelSourceTier::Direct));
    fixture.original_query_clauses.source_tiers[23] = tier(QueryLabelSourceTier::Exact);
    OrRoutePlannerConfig config;
    config.fuzzy_cap_policy = "legacy";
    configure_or_route_planner(config);
    for (uint32_t cap : {12U, 16U, 24U}) {
        fixture.original_query_clauses.branch_caps = {cap};
        const GraphRuntime result = route(fixture);
        std::vector<uint32_t> expected;
        for (uint32_t i = 0; i < std::min(cap, 23U); ++i) expected.push_back(i);
        expected.push_back(23);  // Exact branch does not consume the fuzzy cap.
        require(result.query_filter_labels == expected && result.route_to_clause == expected,
                "singleton DNF cap lost branch identity or exact alternatives");
        require(result.low_query_indices == std::vector<uint32_t>({5}) &&
                    result.high_query_indices.size() == expected.size() - 1,
                "singleton DNF route changed low-frequency dispatch");
        require(result.query_vectors.rows == expected.size() &&
                    result.query_vectors.data.size() == 2 * expected.size(),
                "singleton DNF routes lost query-vector alignment");
    }
    fixture.original_query_clauses.source_tiers[0] = 255;
    require_throws_contains([&] { (void)route(fixture); }, "invalid source tier",
                            "singleton DNF accepted invalid occurrence metadata");
}


void test_compact_singleton_planner_equivalence(const ScopedRouteCaps& caps)
{
    const auto variant = [](GraphRuntime fixture, bool compact) {
        const char* old = std::getenv("GO_FANN_COMPACT_CLAUSE_PLANNER");
        const std::optional<std::string> saved = old ? std::optional<std::string>(old) : std::nullopt;
        const auto restore = [&] {
            if (saved) setenv("GO_FANN_COMPACT_CLAUSE_PLANNER", saved->c_str(), 1);
            else unsetenv("GO_FANN_COMPACT_CLAUSE_PLANNER");
        };
        setenv("GO_FANN_COMPACT_CLAUSE_PLANNER", compact ? "1" : "0", 1);
        try { auto result = route(std::move(fixture)); restore(); return result; }
        catch (...) { restore(); throw; }
    };
    constexpr uint32_t count = 53;
    std::vector<uint32_t> labels(count), branches(count), postings(count), graph(count), low(count);
    std::vector<uint64_t> offsets(count + 1);
    for (uint32_t i = 0; i < count; ++i) {
        labels[i] = (i * 17) % count;
        branches[i] = count - i;
        offsets[i] = i;
        postings[i] = 10 + (i % 5) * 7;
        graph[i] = i % 3 ? 10 + i % 7 : 0;
        low[i] = i % 3 ? 0 : postings[i];
    }
    offsets.back() = count;
    for (const std::string policy : {"legacy", "cost_aware", "coverage_cost", "relevance_coverage_cost"}) {
        OrRoutePlannerConfig config;
        config.order_mode = policy;
        config.fuzzy_cap_policy = policy;
        configure_or_route_planner(config);
        for (uint32_t cap : {0U, 1U, 8U, 12U, 16U, 64U}) {
            for (bool fuzzy : {false, true}) {
                for (uint32_t high_cap : {0U, 3U}) {
                    caps.set(0, high_cap);
                    auto fixture = make_runtime(labels, QueryLabelOp::Or, postings, graph, low);
                    configure_hybrid_clause_fixture(fixture, {0, count}, offsets, labels, branches,
                                                    std::vector<uint8_t>(count, 0));
                    fixture.original_query_clauses.branch_caps = {cap};
                    fixture.original_query_clauses.branch_cap_is_fuzzy = {uint8_t(fuzzy)};
                    for (uint32_t i = 0; i < count; ++i) {
                        fixture.original_query_clauses.source_tiers[i] = i % 5 == 0
                            ? tier(QueryLabelSourceTier::Exact)
                            : tier(i % 3 ? QueryLabelSourceTier::Direct : QueryLabelSourceTier::Fallback);
                        fixture.original_query_clauses.relevance_ranks[i] = i % 7;
                    }
                    const auto reference = variant(fixture, false);
                    const auto compact = variant(fixture, true);
                    require(route_tuples(compact) == route_tuples(reference) &&
                                compact.route_to_clause == reference.route_to_clause &&
                                compact.route_clause_indptr == reference.route_clause_indptr &&
                                compact.route_clause_ids == reference.route_clause_ids &&
                                compact.route_top_branch_ids == reference.route_top_branch_ids &&
                                compact.query_vectors.data == reference.query_vectors.data,
                            "compact singleton planner changed physical route materialization");
                    const auto& a = reference.or_route_planner_profile;
                    const auto& b = compact.or_route_planner_profile;
#define SAME_PROFILE(field) require(a.field == b.field, "compact planner changed profile " #field)
                    SAME_PROFILE(mode);
                    SAME_PROFILE(total_cap);
                    SAME_PROFILE(highfreq_cap);
                    SAME_PROFILE(queries);
                    SAME_PROFILE(original_routes);
                    SAME_PROFILE(emitted_routes);
                    SAME_PROFILE(original_highfreq_routes);
                    SAME_PROFILE(original_lowfreq_routes);
                    SAME_PROFILE(emitted_highfreq_routes);
                    SAME_PROFILE(emitted_lowfreq_routes);
                    SAME_PROFILE(original_exact_source_routes);
                    SAME_PROFILE(emitted_exact_source_routes);
                    SAME_PROFILE(original_direct_source_routes);
                    SAME_PROFILE(emitted_direct_source_routes);
                    SAME_PROFILE(original_fallback_source_routes);
                    SAME_PROFILE(emitted_fallback_source_routes);
                    SAME_PROFILE(retained_relevance_rank_sum);
                    SAME_PROFILE(retained_relevance_rank_max);
                    SAME_PROFILE(reordered_queries);
                    SAME_PROFILE(reordered_routes);
                    SAME_PROFILE(capped_queries);
                    SAME_PROFILE(capped_routes);
                    SAME_PROFILE(total_cap_capped_routes);
                    SAME_PROFILE(highfreq_cap_capped_routes);
                    SAME_PROFILE(manifest_cap_queries);
                    SAME_PROFILE(manifest_cap_max);
                    SAME_PROFILE(original_exact_lowfreq_rows);
                    SAME_PROFILE(emitted_exact_lowfreq_rows);
                    SAME_PROFILE(original_highfreq_cost_proxy);
                    SAME_PROFILE(emitted_highfreq_cost_proxy);
                    SAME_PROFILE(original_exact_label_cardinality_sum);
                    SAME_PROFILE(emitted_exact_label_cardinality_sum);
                    SAME_PROFILE(sketch_labels);
                    SAME_PROFILE(sketch_hashes);
                    SAME_PROFILE(sketch_bytes);
                    SAME_PROFILE(sketch_sidecar_bytes_read);
                    SAME_PROFILE(sketch_build_rows);
                    SAME_PROFILE(retained_zero_marginal_routes);
                    SAME_PROFILE(exact_cardinality_used);
                    SAME_PROFILE(estimated_union_coverage_sum);
                    SAME_PROFILE(retained_estimated_marginal_gain_sum);
                    SAME_PROFILE(original_predicted_latency_us_sum);
                    SAME_PROFILE(emitted_predicted_latency_us_sum);
                    SAME_PROFILE(emitted_quality_sum);
                    SAME_PROFILE(emitted_utility_sum);
#undef SAME_PROFILE
                }
            }
        }
    }
    caps.clear();
}

void test_misaligned_occurrence_metadata_is_rejected()
{
    GraphRuntime fixture = make_runtime(
        {0, 1}, QueryLabelOp::Or, {10, 20}, {0, 0}, {10, 20});
    fixture.original_query_labels_matrix.source_tiers = {
        tier(QueryLabelSourceTier::Direct)};
    fixture.original_query_labels_matrix.relevance_ranks = {0, 0};
    configure_or_route_planner("relevance_coverage_cost");
    bool failed = false;
    try {
        (void)route(fixture);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("CSR-aligned") != std::string::npos;
    }
    require(failed, "misaligned occurrence metadata was accepted");
}

void test_invalid_occurrence_metadata_is_rejected_before_routing()
{
    const auto require_rejected = [](GraphRuntime fixture,
                                     const std::string& expected_message,
                                     const std::string& context) {
        bool failed = false;
        try {
            (void)route(std::move(fixture));
        } catch (const std::runtime_error& error) {
            failed = std::string(error.what()).find(expected_message) != std::string::npos;
        }
        require(failed, context);
    };

    GraphRuntime invalid_tier = make_runtime(
        {0},
        QueryLabelOp::And,
        {10},
        {0},
        {10},
        4,
        {},
        {99},
        {0});
    require_rejected(std::move(invalid_tier),
                     "invalid source tier",
                     "invalid source tier on an AND row was accepted");

    GraphRuntime invalid_exact_rank = make_runtime(
        {0},
        QueryLabelOp::Or,
        {10},
        {0},
        {10},
        4,
        {},
        {tier(QueryLabelSourceTier::Exact)},
        {1});
    require_rejected(std::move(invalid_exact_rank),
                     "requires relevance rank zero",
                     "nonzero exact relevance rank was accepted");

    GraphRuntime invalid_fallback_rank = make_runtime(
        {0},
        QueryLabelOp::Or,
        {10},
        {0},
        {10},
        4,
        {},
        {tier(QueryLabelSourceTier::Fallback)},
        {0});
    require_rejected(std::move(invalid_fallback_rank),
                     "one-based relevance rank",
                     "zero fallback relevance rank was accepted");

    GraphRuntime misaligned_row_cache = make_runtime(
        {0, 1},
        QueryLabelOp::Or,
        {10, 20},
        {0, 0},
        {10, 20},
        4,
        {},
        {tier(QueryLabelSourceTier::Exact), tier(QueryLabelSourceTier::Exact)},
        {0, 0});
    misaligned_row_cache.original_query_labels_matrix.row_labels = {{1, 0}};
    require_rejected(std::move(misaligned_row_cache),
                     "row label cache is not aligned",
                     "row cache and CSR label permutation mismatch was accepted");
}

void test_group_union_is_not_original_label_coverage(const ScopedRouteCaps& caps)
{
    GraphRuntime fixture = make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {5, 0, 4}, {5, 0, 4}, {0, 0, 0});
    fixture.group_layout.label_to_group = {0, 0, 2};
    fixture.group_layout.groups = {{0, 1}, {}, {2}};
    fixture.metadata.label_filter_bit = {1, 2, 1};
    fixture.metadata.label_offset = {0, 5, 5};
    fixture.metadata.row_filter_mask = {1, 1, 2, 2, 2, 1, 1, 1, 1};
    fixture.metadata.original_label_cardinality.clear();

    configure_or_route_planner("coverage_cost");
    caps.set(2, 2);
    const GraphRuntime runtime = route(fixture);
    caps.clear();

    require(runtime.metadata.original_label_cardinality ==
                std::vector<uint32_t>({2, 3, 4}),
            "group filter masks did not produce exact original-label cardinalities");
    require(runtime.query_filter_labels == std::vector<uint32_t>({2, 1}),
            "coverage retention used group union size instead of original-label cardinality");
}

void test_lowfreq_grouped_and_singleton_cardinality_is_cached_once()
{
    GraphRuntime fixture = make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {3, 0, 2}, {0, 0, 0}, {3, 0, 2});
    fixture.group_layout.label_to_group = {0, 0, 2};
    fixture.group_layout.groups = {{0, 1}, {}, {2}};
    fixture.metadata.label_filter_bit = {1, 2, 1};
    fixture.metadata.original_label_cardinality.clear();
    fixture.lowfreq_index.label_offset = {0, 3, 3};
    fixture.lowfreq_index.row_filter_mask = {1, 3, 2, 1, 1};

    configure_or_route_planner("coverage_cost");
    validate_and_prepare_queries(fixture, 2, "greedy_min_posting");
    require(fixture.metadata.original_label_cardinality ==
                std::vector<uint32_t>({2, 2, 2}),
            "low-frequency grouped masks or singleton posting length produced wrong cardinality");
    require(fixture.or_route_planner_profile.original_exact_label_cardinality_sum == 6,
            "overlap-counting profile did not sum each route cardinality");

    validate_and_prepare_queries(fixture, 2, "greedy_min_posting");
    require(fixture.metadata.original_label_cardinality ==
                std::vector<uint32_t>({2, 2, 2}),
            "reentrant coverage planning accumulated cached cardinalities twice");
    require(fixture.or_route_planner_profile.original_exact_label_cardinality_sum == 6,
            "reentrant coverage profile was not reset before recounting routes");
}

void test_failed_cardinality_derivation_does_not_poison_retry()
{
    GraphRuntime fixture = make_runtime(
        {0, 1}, QueryLabelOp::Or, {2, 0}, {2, 0}, {0, 0});
    fixture.group_layout.label_to_group = {0, 0};
    fixture.group_layout.groups = {{0, 1}, {}};
    fixture.metadata.label_filter_bit = {1, 2};
    fixture.metadata.label_offset = {0, 2};
    fixture.metadata.row_filter_mask = {3, 0};
    fixture.metadata.original_label_cardinality.clear();

    configure_or_route_planner("coverage_cost");
    bool failed = false;
    try {
        validate_and_prepare_queries(fixture, 2, "greedy_min_posting");
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("invalid grouped") != std::string::npos;
    }
    require(failed, "invalid grouped filter mask did not fail cardinality derivation");
    require(fixture.metadata.original_label_cardinality.empty(),
            "failed cardinality derivation left a partial cache");

    fixture.metadata.row_filter_mask[1] = 1;
    validate_and_prepare_queries(fixture, 2, "greedy_min_posting");
    require(fixture.metadata.original_label_cardinality ==
                std::vector<uint32_t>({2, 1}),
            "cardinality retry did not derive exact values after repairing masks");
}

void test_uncapped_coverage_cost_preserves_multiset()
{
    const GraphRuntime fixture = make_runtime(
        {0, 1, 2}, QueryLabelOp::Or, {8, 2, 99}, {8, 2, 0}, {0, 0, 1});
    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(fixture);
    configure_or_route_planner("coverage_cost");
    const GraphRuntime coverage_cost = route(fixture);
    configure_or_route_planner("relevance_coverage_cost");
    const GraphRuntime relevance_coverage_cost = route(fixture);

    require(sorted_route_tuples(coverage_cost) == sorted_route_tuples(legacy),
            "uncapped coverage_cost changed the route multiset or metadata");
    require(sorted_route_tuples(relevance_coverage_cost) == sorted_route_tuples(legacy),
            "uncapped relevance_coverage_cost changed the route multiset or metadata");
}

void test_grouped_coverage_metadata_is_required()
{
    GraphRuntime fixture = make_runtime(
        {0, 1}, QueryLabelOp::Or, {5, 0}, {5, 0}, {0, 0});
    fixture.group_layout.label_to_group = {0, 0};
    fixture.group_layout.groups = {{0, 1}, {}};
    fixture.metadata.label_filter_bit = {1, 2};
    fixture.metadata.original_label_cardinality.clear();
    configure_or_route_planner("coverage_cost");
    bool failed = false;
    try {
        (void)route(fixture);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("filter masks") != std::string::npos;
    }
    require(failed,
            "coverage_cost silently accepted grouped coverage without authoritative masks");
}

void test_caps_apply_after_cost_ordering(const ScopedRouteCaps& caps)
{
    configure_or_route_planner("cost_aware");
    caps.set(3, 1);
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2, 3, 4},
        QueryLabelOp::Or,
        {100, 1, 50, 40, 30},
        {2, 1, 0, 0, 0},
        {0, 0, 12, 10, 14}));
    caps.clear();

    require(runtime.query_filter_labels == std::vector<uint32_t>({1, 3, 2}),
            "OR caps were not applied after cost-aware ordering");
    const OrRoutePlannerProfile& profile = runtime.or_route_planner_profile;
    require(profile.original_routes == 5 && profile.emitted_routes == 3,
            "cap route counters are incorrect");
    require(profile.capped_queries == 1 && profile.capped_routes == 2 &&
                profile.highfreq_cap_capped_routes == 1 &&
                profile.total_cap_capped_routes == 1,
            "cap reason counters are incorrect");
    require(profile.original_exact_lowfreq_rows == 36 &&
                profile.emitted_exact_lowfreq_rows == 22 &&
                profile.original_highfreq_cost_proxy == 12 &&
                profile.emitted_highfreq_cost_proxy == 4,
            "cap cost profiles do not distinguish original and emitted work");
    require(profile.reordered_queries == 1 && profile.reordered_routes > 0,
            "cap fixture did not exercise cost-aware reordering");
}

void test_coverage_caps_skip_blocked_highfreq_and_fill_total(const ScopedRouteCaps& caps)
{
    configure_or_route_planner("coverage_cost");
    caps.set(3, 1);
    const GraphRuntime runtime = route(make_runtime(
        {0, 1, 2, 3, 4},
        QueryLabelOp::Or,
        {100, 90, 80, 70, 60},
        {100, 90, 0, 0, 0},
        {0, 0, 80, 70, 60}));
    caps.clear();

    require(runtime.query_filter_labels == std::vector<uint32_t>({3, 2, 0}),
            "coverage caps did not skip a blocked graph route and fill with postings");
    const OrRoutePlannerProfile& profile = runtime.or_route_planner_profile;
    require(profile.original_routes == 5 && profile.emitted_routes == 3 &&
                profile.original_highfreq_routes == 2 && profile.emitted_highfreq_routes == 1,
            "coverage caps emitted the wrong total or high-frequency route counts");
    require(profile.capped_routes == 2 && profile.highfreq_cap_capped_routes == 1 &&
                profile.total_cap_capped_routes == 1,
            "coverage caps attributed dropped routes to the wrong constraints");
    require(profile.emitted_exact_label_cardinality_sum == 250,
            "coverage caps retained the wrong exact-cardinality set");
}

void test_legacy_alias_caps_and_cross_run_reset(const ScopedRouteCaps& caps)
{
    const GraphRuntime fixture = make_runtime(
        {0, 1, 2, 3, 4},
        QueryLabelOp::Or,
        {100, 1, 50, 40, 30},
        {2, 1, 0, 0, 0},
        {0, 0, 12, 10, 14},
        4,
        {},
        {tier(QueryLabelSourceTier::Direct),
         tier(QueryLabelSourceTier::Direct),
         tier(QueryLabelSourceTier::Direct),
         tier(QueryLabelSourceTier::Direct),
         tier(QueryLabelSourceTier::Direct)},
        {0, 0, 0, 0, 0});

    caps.set(3, 1);
    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(fixture);
    configure_or_route_planner("manifest");
    const GraphRuntime manifest = route(fixture);
    caps.clear();

    require(legacy.query_filter_labels == std::vector<uint32_t>({0, 2, 3}),
            "legacy caps no longer preserve historical manifest-order output");
    require(route_tuples(manifest) == route_tuples(legacy),
            "manifest alias differs from legacy mode");
    require(legacy.or_route_planner_profile.capped_routes == 2 &&
                legacy.or_route_planner_profile.total_cap_capped_routes == 1 &&
                legacy.or_route_planner_profile.highfreq_cap_capped_routes == 1,
            "legacy cap accounting is incorrect");

    configure_or_route_planner("relevance_coverage_cost");
    GraphRuntime reused = fixture;
    validate_and_prepare_queries(reused, 2, "greedy_min_posting");
    configure_or_route_planner("");
    validate_and_prepare_queries(reused, 2, "greedy_min_posting");
    validate_graph_runtime_query_metadata(reused, "or_route_planner_reset_test");
    require(reused.query_filter_labels == std::vector<uint32_t>({0, 1, 2, 3, 4}),
            "empty mode did not reset the global planner selection to legacy");
    require(reused.or_route_planner_profile.mode == OrRouteOrderMode::Legacy &&
                reused.or_route_planner_profile.queries == 1 &&
                reused.or_route_planner_profile.original_routes == 5 &&
                reused.or_route_planner_profile.emitted_routes == 5 &&
                reused.or_route_planner_profile.original_direct_source_routes == 5 &&
                reused.or_route_planner_profile.emitted_direct_source_routes == 5 &&
                reused.or_route_planner_profile.original_exact_label_cardinality_sum == 0 &&
                reused.or_route_planner_profile.emitted_exact_label_cardinality_sum == 0,
            "planner profile was not reset between runtime preparations");
}

void test_marginal_model_contract_rejections()
{
    const GraphRuntime runtime = make_overlap_runtime();
    const std::string invalid_path = temporary_config_path("_invalid_model.json");
    ModelFixtureOptions invalid;
    invalid.validated = false;
    write_model_fixture(invalid_path, runtime, 16, 123, invalid);
    bool failed = false;
    try {
        configure_marginal_fixture(runtime, invalid_path, 16, 123);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("validated=true") != std::string::npos;
    }
    require(failed, "validated=false model was accepted");

    invalid.validated = true;
    invalid.add_unknown_field = true;
    write_model_fixture(invalid_path, runtime, 16, 123, invalid);
    failed = false;
    try {
        configure_marginal_fixture(runtime, invalid_path, 16, 123);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("unknown field") != std::string::npos;
    }
    require(failed, "model with unknown fields was accepted");

    ModelFixtureOptions missing_base_weight;
    missing_base_weight.omit_base_coverage_weight = true;
    write_model_fixture(invalid_path, runtime, 16, 123, missing_base_weight);
    failed = false;
    try {
        configure_marginal_fixture(runtime, invalid_path, 16, 123);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("base_coverage_weight") != std::string::npos;
    }
    require(failed, "model without base_coverage_weight was accepted");

    ModelFixtureOptions zero_base_weight;
    zero_base_weight.base_coverage_weight = 0.0;
    write_model_fixture(invalid_path, runtime, 16, 123, zero_base_weight);
    failed = false;
    try {
        configure_marginal_fixture(runtime, invalid_path, 16, 123);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("finite valid range") != std::string::npos;
    }
    require(failed, "non-positive base_coverage_weight was accepted");

    ModelFixtureOptions zero_marginal_weight;
    zero_marginal_weight.marginal_coverage_weight = 0.0;
    write_model_fixture(invalid_path, runtime, 16, 123, zero_marginal_weight);
    configure_marginal_fixture(runtime, invalid_path, 16, 123);

    OrRoutePlannerConfig missing;
    missing.order_mode = "marginal_accuracy_cost";
    missing.allow_runtime_sketch_fallback = true;
    failed = false;
    try {
        configure_or_route_planner(missing);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("model path") != std::string::npos;
    }
    require(failed, "marginal planner without a model was accepted");

    OrRoutePlannerConfig invalid_k;
    invalid_k.order_mode = "legacy";
    invalid_k.sketch_k = 3;
    failed = false;
    try {
        configure_or_route_planner(invalid_k);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("[4, 4096]") != std::string::npos;
    }
    require(failed, "out-of-range sketch K was accepted");

    ModelFixtureOptions negative;
    negative.lowfreq_per_row_us = -1.0;
    negative.add_unknown_field = false;
    write_model_fixture(invalid_path, runtime, 16, 123, negative);
    failed = false;
    try {
        configure_marginal_fixture(runtime, invalid_path, 16, 123);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("finite valid range") != std::string::npos;
    }
    require(failed, "negative calibrated cost coefficient was accepted");
    std::remove(invalid_path.c_str());
    configure_or_route_planner("legacy");
}

void test_marginal_model_size_and_diagnostics_gates()
{
    const GraphRuntime runtime = make_overlap_runtime();
    const std::string path = temporary_config_path("_diagnostics_model.json");
    const auto require_rejected = [&](const ModelFixtureOptions& options,
                                      const std::string& expected,
                                      const std::string& context) {
        write_model_fixture(path, runtime, 16, 123, options);
        bool failed = false;
        try {
            configure_marginal_fixture(runtime, path, 16, 123);
        } catch (const std::runtime_error& error) {
            failed = std::string(error.what()).find(expected) != std::string::npos;
        }
        require(failed, context);
    };

    ModelFixtureOptions options;
    options.training_sample_count = 999;
    require_rejected(options, "training_sample_count", "undersized training samples were accepted");
    options = ModelFixtureOptions{};
    options.training_query_count = 99;
    require_rejected(options, "training_query_count", "undersized training queries were accepted");
    options = ModelFixtureOptions{};
    options.validation_sample_count = 249;
    require_rejected(options, "validation_sample_count", "undersized validation samples were accepted");
    options = ModelFixtureOptions{};
    options.validation_query_count = 24;
    require_rejected(options, "validation_query_count", "undersized validation queries were accepted");
    options = ModelFixtureOptions{};
    options.calibration_run_count = 2;
    require_rejected(options, "calibration_run_count", "fewer than three runs were accepted");

    options = ModelFixtureOptions{};
    options.utility_search_candidates = 1;
    require_rejected(options,
                     "utility_search_candidates",
                     "single-candidate utility search was accepted");
    options = ModelFixtureOptions{};
    options.heldout_gt_coverage = 800;
    options.heldout_gt_coverage_ratio = 0.8;
    options.min_required_gt_coverage_ratio = 0.9;
    require_rejected(options,
                     "below the required minimum",
                     "held-out coverage below its gate was accepted");
    options = ModelFixtureOptions{};
    options.heldout_predicted_latency_us = 110.0;
    options.heldout_predicted_latency_ratio = 1.1;
    options.max_allowed_latency_ratio = 1.05;
    require_rejected(options,
                     "exceeds the allowed maximum",
                     "held-out latency above its gate was accepted");
    options = ModelFixtureOptions{};
    options.heldout_gt_coverage = 0;
    options.heldout_baseline_gt_coverage = 0;
    require_rejected(options,
                     "below its minimum",
                     "zero held-out baseline coverage was accepted");
    options = ModelFixtureOptions{};
    options.heldout_gt_coverage_ratio = 0.95;
    options.min_required_gt_coverage_ratio = 0.9;
    require_rejected(options,
                     "numerator/baseline",
                     "inconsistent held-out coverage ratio was accepted");
    options = ModelFixtureOptions{};
    options.selection_route_cap = 8;
    options.selection_highfreq_cap = 9;
    require_rejected(options,
                     "exceeds selection_route_cap",
                     "inconsistent selection caps were accepted");

    OrRoutePlannerConfig missing_run_id;
    missing_run_id.calibration_trace_path = path + ".trace";
    bool failed = false;
    try {
        configure_or_route_planner(missing_run_id);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("nonempty calibration run id") !=
                 std::string::npos;
    }
    require(failed, "calibration trace without a run id was accepted");

    std::remove(path.c_str());
    configure_or_route_planner("legacy");
}

void test_marginal_model_runtime_cap_binding(const ScopedRouteCaps& caps)
{
    const GraphRuntime fixture = make_overlap_runtime();
    const std::string path = temporary_config_path("_cap_binding_model.json");
    ModelFixtureOptions options;
    options.selection_route_cap = 2;
    options.selection_highfreq_cap = 1;
    write_model_fixture(path, fixture, 16, 123, options);
    configure_marginal_fixture(fixture, path, 16, 123);

    caps.set(1, 1);
    bool failed = false;
    try {
        (void)route(fixture);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("runtime caps do not match") !=
                 std::string::npos;
    }
    require(failed, "calibrated model was accepted under different runtime caps");

    caps.set(2, 1);
    (void)route(fixture);
    caps.clear();
    std::remove(path.c_str());
    configure_or_route_planner("legacy");
}

void test_or_route_calibration_trace_mapping_order_and_reentry()
{
    const std::string trace_path = temporary_config_path("_candidate_trace.jsonl");
    OrRoutePlannerConfig config;
    config.order_mode = "cost_aware";
    config.calibration_trace_path = trace_path;
    config.calibration_run_id = "trace-run-1";
    configure_or_route_planner(config);

    GraphRuntime runtime = make_mixed_trace_runtime();
    validate_and_prepare_queries(runtime, 2, "greedy_min_posting");
    validate_graph_runtime_query_metadata(runtime, "or_route_trace_test");
    require(runtime.route_to_original_query == std::vector<uint32_t>({0, 1, 1}) &&
                runtime.query_filter_labels == std::vector<uint32_t>({0, 1, 2}),
            "mixed fixture route ids are not in backend runtime order");

    const std::vector<nlohmann::json> first = read_jsonl(trace_path);
    require(first.size() == 2,
            "uncapped trace did not contain every routable OR candidate exactly once");
    require(first[0].size() == 14 &&
                first[0].at("schema") == "flashanns.or_route_candidate_trace" &&
                first[0].at("version") == 1 &&
                first[0].at("run_id") == "trace-run-1" &&
                first[0].at("original_query_id") == 1 &&
                first[0].at("routed_route_id") == 1 &&
                first[0].at("label") == 1 &&
                first[0].at("tier") == "direct" &&
                first[0].at("rank") == 0 &&
                first[0].at("exact_label_cardinality") == 2 &&
                first[0].at("route_type") == "low" &&
                first[0].at("lowfreq_rows") == 2 &&
                first[0].at("highfreq_graph_rows") == 0 &&
                first[0].at("highfreq_degree") == 0 &&
                first[0].at("highfreq_work") == 0,
            "low-frequency candidate trace row is incorrect");
    require(first[1].at("original_query_id") == 1 &&
                first[1].at("routed_route_id") == 2 &&
                first[1].at("label") == 2 &&
                first[1].at("tier") == "fallback" &&
                first[1].at("rank") == 1 &&
                first[1].at("exact_label_cardinality") == 4 &&
                first[1].at("route_type") == "high" &&
                first[1].at("lowfreq_rows") == 0 &&
                first[1].at("highfreq_graph_rows") == 4 &&
                first[1].at("highfreq_degree") == 4 &&
                first[1].at("highfreq_work") == 16,
            "high-frequency candidate trace row is incorrect");

    validate_and_prepare_queries(runtime, 2, "greedy_min_posting");
    const std::vector<nlohmann::json> second = read_jsonl(trace_path);
    require(second == first,
            "reentrant query preparation appended or reordered calibration trace rows");

    std::remove(trace_path.c_str());
    configure_or_route_planner("legacy");
}

void test_or_route_calibration_trace_cap_conflict(const ScopedRouteCaps& caps)
{
    const std::string trace_path = temporary_config_path("_capped_trace.jsonl");
    OrRoutePlannerConfig config;
    config.order_mode = "cost_aware";
    config.calibration_trace_path = trace_path;
    config.calibration_run_id = "capped-run";
    configure_or_route_planner(config);
    caps.set(1, 1);
    bool failed = false;
    try {
        (void)route(make_runtime(
            {0, 1}, QueryLabelOp::Or, {3, 2}, {0, 0}, {3, 2}));
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("to both be 0") != std::string::npos;
    }
    caps.clear();
    require(failed, "calibration trace accepted active OR route caps");
    std::remove(trace_path.c_str());
    configure_or_route_planner("legacy");
}

void test_or_route_calibration_trace_bypasses_and_and_range()
{
    const std::string trace_path = temporary_config_path("_non_or_trace.jsonl");
    OrRoutePlannerConfig config;
    config.order_mode = "cost_aware";
    config.calibration_trace_path = trace_path;
    config.calibration_run_id = "non-or-run";
    configure_or_route_planner(config);

    (void)route(make_runtime(
        {0, 1}, QueryLabelOp::And, {5, 100}, {100, 1}, {0, 0}));
    require(read_jsonl(trace_path).empty(), "AND route leaked into OR calibration trace");

    GraphRuntime range = make_runtime(
        {}, QueryLabelOp::And, {10}, {10}, {0});
    range.base_vector_metadata.rows = 10;
    range.original_query_ranges.range_count = 1;
    range.original_query_ranges.modes = {3};
    range.original_query_ranges.lower_bounds = {0.0f};
    range.original_query_ranges.upper_bounds = {1.0f};
    range.label_intervals.has_interval = {1};
    range.label_intervals.range_indices = {0};
    range.label_intervals.level_indices = {0};
    range.label_intervals.bin_indices = {0};
    range.label_intervals.lower_bounds = {0.0f};
    range.label_intervals.upper_bounds = {1.0f};
    (void)route(range);
    require(read_jsonl(trace_path).empty(), "range route leaked into OR calibration trace");

    std::remove(trace_path.c_str());
    configure_or_route_planner("legacy");
}

void test_fallback_utility_is_not_cheapest_first(const ScopedRouteCaps& caps)
{
    GraphRuntime fixture = make_overlap_runtime();
    fixture.original_query_labels_matrix.source_tiers = {
        tier(QueryLabelSourceTier::Fallback),
        tier(QueryLabelSourceTier::Fallback),
        tier(QueryLabelSourceTier::Fallback),
    };
    fixture.original_query_labels_matrix.relevance_ranks = {1, 1, 2};
    const std::string model_path = temporary_config_path("_fallback_utility_model.json");
    ModelFixtureOptions options;
    options.selection_route_cap = 2;
    options.selection_highfreq_cap = 2;
    write_model_fixture(model_path, fixture, 16, 123, options);
    configure_marginal_fixture(fixture, model_path, 16, 123);
    caps.set(2, 2);
    const GraphRuntime runtime = route(fixture);
    caps.clear();
    require(runtime.query_filter_labels == std::vector<uint32_t>({2, 0}),
            "fallback utility collapsed to cheapest-first retention");
    require(runtime.or_route_planner_profile.emitted_fallback_source_routes == 2 &&
                runtime.or_route_planner_profile.retained_relevance_rank_sum == 3,
            "fallback rank/retention diagnostics are incorrect");
    std::remove(model_path.c_str());
}

void test_marginal_overlap_beats_cardinality_only(const ScopedRouteCaps& caps)
{
    const GraphRuntime fixture = make_overlap_runtime();
    caps.set(2, 2);
    configure_or_route_planner("coverage_cost");
    const GraphRuntime cardinality = route(fixture);
    require(cardinality.query_filter_labels == std::vector<uint32_t>({1, 0}),
            "cardinality-only control did not select the overlapping largest labels");

    const std::string model_path = temporary_config_path("_overlap_model.json");
    ModelFixtureOptions options;
    options.selection_route_cap = 2;
    options.selection_highfreq_cap = 2;
    write_model_fixture(model_path, fixture, 16, 123, options);
    configure_marginal_fixture(fixture, model_path, 16, 123);
    const GraphRuntime marginal = route(fixture);
    caps.clear();

    require(marginal.query_filter_labels == std::vector<uint32_t>({2, 0}),
            "marginal overlap planner did not retain the disjoint smaller label");
    const OrRoutePlannerProfile& profile = marginal.or_route_planner_profile;
    require(profile.sketch_labels == 3 && profile.sketch_hashes == 12 &&
                profile.sketch_build_rows == 12,
            "runtime active-label sketch profile is incorrect");
    require(std::abs(profile.estimated_union_coverage_sum - 8.0) < 1e-9 &&
                std::abs(profile.retained_estimated_marginal_gain_sum - 8.0) < 1e-9 &&
                profile.retained_zero_marginal_routes == 0,
            "marginal union/gain profile is incorrect");
    require(profile.calibration_id == "synthetic-heldout-v1" &&
                profile.original_predicted_latency_us_sum >
                    profile.emitted_predicted_latency_us_sum,
            "calibration/latency diagnostics are missing");
    std::remove(model_path.c_str());
}

void test_base_coverage_and_marginal_overlap_are_separate(const ScopedRouteCaps& caps)
{
    const GraphRuntime fixture = make_overlap_runtime();
    ModelFixtureOptions options;
    options.base_coverage_weight = 10.0;
    options.marginal_coverage_weight = 0.001;
    options.selection_route_cap = 2;
    options.selection_highfreq_cap = 2;
    const std::string model_path = temporary_config_path("_base_marginal_model.json");
    write_model_fixture(model_path, fixture, 16, 123, options);
    configure_marginal_fixture(fixture, model_path, 16, 123);
    caps.set(2, 2);
    const GraphRuntime base_preserved = route(fixture);
    require(base_preserved.query_filter_labels == std::vector<uint32_t>({1, 0}),
            "base coverage did not preserve the overlapping high-cardinality candidate");

    options.marginal_coverage_weight = 10.0;
    write_model_fixture(model_path, fixture, 16, 123, options);
    configure_marginal_fixture(fixture, model_path, 16, 123);
    const GraphRuntime overlap_aware = route(fixture);
    caps.clear();
    require(overlap_aware.query_filter_labels == std::vector<uint32_t>({2, 0}),
            "marginal coverage bonus did not distinguish the disjoint candidate");
    std::remove(model_path.c_str());
}

void test_marginal_hard_relevance_guard(const ScopedRouteCaps& caps)
{
    GraphRuntime fixture = make_overlap_runtime();
    fixture.original_query_labels_matrix.source_tiers = {
        tier(QueryLabelSourceTier::Direct),
        tier(QueryLabelSourceTier::Fallback),
        tier(QueryLabelSourceTier::Fallback),
    };
    fixture.original_query_labels_matrix.relevance_ranks = {0, 1, 2};
    ModelFixtureOptions options;
    options.direct_prior = 0.1;
    options.fallback_prior = 100.0;
    options.marginal_coverage_weight = 0.001;
    options.cost_weight = 1.0;
    options.selection_route_cap = 1;
    options.selection_highfreq_cap = 1;
    const std::string model_path = temporary_config_path("_guard_model.json");
    write_model_fixture(model_path, fixture, 16, 123, options);
    configure_marginal_fixture(fixture, model_path, 16, 123);
    caps.set(1, 1);
    const GraphRuntime guarded = route(fixture);
    caps.clear();
    require(guarded.query_filter_labels == std::vector<uint32_t>({0}),
            "cheap fallback displaced a guarded direct route");
    require(guarded.or_route_planner_profile.emitted_direct_source_routes == 1 &&
                guarded.or_route_planner_profile.emitted_fallback_source_routes == 0,
            "hard relevance guard profile is incorrect");
    std::remove(model_path.c_str());
}

void test_marginal_uncapped_small_k_determinism_and_reentry()
{
    GraphRuntime fixture = make_overlap_runtime();
    const std::string model_path = temporary_config_path("_small_k_model.json");
    write_model_fixture(model_path, fixture, 4, 987);
    configure_marginal_fixture(fixture, model_path, 4, 987);
    const GraphRuntime first = route(fixture);
    const GraphRuntime second = route(fixture);
    require(first.query_filter_labels == std::vector<uint32_t>({2, 1, 0}),
            "uncapped marginal planner did not schedule by predicted latency");
    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(make_overlap_runtime());
    require(sorted_route_tuples(first) == sorted_route_tuples(legacy),
            "uncapped marginal planner changed the complete route multiset");
    require(route_tuples(first) == route_tuples(second) &&
                first.or_route_planner_profile.estimated_union_coverage_sum ==
                    second.or_route_planner_profile.estimated_union_coverage_sum,
            "small-K marginal planning is not deterministic");

    configure_marginal_fixture(fixture, model_path, 4, 987);
    validate_and_prepare_queries(fixture, 2, "greedy_min_posting");
    const OrRoutePlannerProfile first_profile = fixture.or_route_planner_profile;
    validate_and_prepare_queries(fixture, 2, "greedy_min_posting");
    require(fixture.or_route_planner_profile.sketch_labels == first_profile.sketch_labels &&
                fixture.or_route_planner_profile.sketch_build_rows == first_profile.sketch_build_rows &&
                fixture.or_route_planner_profile.emitted_routes == first_profile.emitted_routes,
            "reentrant marginal preparation accumulated profile state");
    std::remove(model_path.c_str());
}

void test_marginal_sidecar_loader_contracts(const ScopedRouteCaps& caps)
{
    const GraphRuntime fixture = make_overlap_runtime();
    const std::vector<std::vector<uint32_t>> samples = {
        {0, 1, 2, 3, 4}, {0, 1, 2, 3}, {5, 6, 7}};
    const std::string model_path = temporary_config_path("_sidecar_model.json");
    const std::string valid_path = temporary_config_path("_valid_sketch.bin");
    ModelFixtureOptions options;
    options.selection_route_cap = 2;
    options.selection_highfreq_cap = 2;
    write_model_fixture(model_path, fixture, 16, 123, options);
    write_sketch_sidecar_fixture(valid_path, fixture, samples, 16, 123);
    configure_marginal_fixture(fixture, model_path, 16, 123, valid_path);
    caps.set(2, 2);
    const GraphRuntime valid = route(fixture);
    require(valid.query_filter_labels == std::vector<uint32_t>({2, 0}) &&
                valid.or_route_planner_profile.sketch_build_rows == 0 &&
                valid.or_route_planner_profile.sketch_bytes > 0,
            "valid sidecar did not serve active-label sketches read-only");
    caps.clear();
    options.selection_route_cap = 0;
    options.selection_highfreq_cap = 0;
    write_model_fixture(model_path, fixture, 16, 123, options);

    const auto require_sidecar_failure = [&](const std::string& path,
                                             const std::string& expected,
                                             const std::string& context) {
        configure_marginal_fixture(fixture, model_path, 16, 123, path);
        bool failed = false;
        try {
            (void)route(fixture);
        } catch (const std::runtime_error& error) {
            failed = std::string(error.what()).find(expected) != std::string::npos;
        }
        require(failed, context);
    };

    const std::string bad_points = temporary_config_path("_bad_points_sketch.bin");
    write_sketch_sidecar_fixture(bad_points, fixture, samples, 16, 123, 9);
    require_sidecar_failure(bad_points, "does not match", "sidecar point mismatch was accepted");
    const std::string bad_layout = temporary_config_path("_bad_layout_sketch.bin");
    write_sketch_sidecar_fixture(bad_layout, fixture, samples, 16, 123, 0, 12345);
    require_sidecar_failure(bad_layout, "does not match", "sidecar group layout mismatch was accepted");
    const std::string truncated = temporary_config_path("_truncated_sketch.bin");
    write_sketch_sidecar_fixture(truncated, fixture, samples, 16, 123, 0, 0, true);
    require_sidecar_failure(truncated, "exact file size", "truncated sidecar was accepted");
    const std::string bad_offsets = temporary_config_path("_bad_offsets_sketch.bin");
    write_sketch_sidecar_fixture(bad_offsets, fixture, samples, 16, 123, 0, 0, false, true);
    require_sidecar_failure(bad_offsets, "invalid offsets", "invalid active offsets were accepted");

    OrRoutePlannerConfig verify_config;
    verify_config.order_mode = "marginal_accuracy_cost";
    verify_config.accuracy_cost_model_path = model_path;
    verify_config.sketch_sidecar_path = valid_path;
    verify_config.sketch_k = 16;
    verify_config.sketch_hash_seed = 123;
    verify_config.verify_sketch_payload_checksum = true;
    configure_or_route_planner(verify_config);
    bool checksum_failed = false;
    try {
        (void)route(fixture);
    } catch (const std::runtime_error& error) {
        checksum_failed = std::string(error.what()).find("checksum mismatch") != std::string::npos;
    }
    require(checksum_failed, "sidecar payload checksum failure was not detected");

    for (const std::string& path : {model_path, valid_path, bad_points, bad_layout,
                                    truncated, bad_offsets}) {
        std::remove(path.c_str());
    }
}

void test_large_runtime_sketch_fallback_is_rejected()
{
    GraphRuntime fixture = make_overlap_runtime();
    fixture.base_vector_metadata.rows = 1001;
    const std::string model_path = temporary_config_path("_large_fallback_model.json");
    write_model_fixture(model_path, fixture, 16, 123);
    configure_marginal_fixture(fixture, model_path, 16, 123);
    bool failed = false;
    try {
        (void)route(fixture);
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()).find("small-fixture row limit") != std::string::npos;
    }
    require(failed, "large runtime attempted to build sketches from runtime postings");
    std::remove(model_path.c_str());
}

void test_marginal_mode_does_not_change_and_routing()
{
    GraphRuntime fixture = make_runtime(
        {0, 1}, QueryLabelOp::And, {5, 100}, {100, 1}, {0, 0});
    fixture.base_vector_metadata.rows = 100;
    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(fixture);
    const std::string model_path = temporary_config_path("_and_model.json");
    write_model_fixture(model_path, fixture, 4, 123);
    configure_marginal_fixture(fixture, model_path, 4, 123);
    const GraphRuntime marginal = route(fixture);
    require(route_tuples(marginal) == route_tuples(legacy) &&
                marginal.or_route_planner_profile.queries == 0 &&
                marginal.or_route_planner_profile.sketch_labels == 0,
            "marginal OR mode changed AND routing or built sketches");
    std::remove(model_path.c_str());
}

void test_profile_cost_saturation()
{
    configure_or_route_planner("cost_aware");
    const uint32_t max_u32 = std::numeric_limits<uint32_t>::max();
    const GraphRuntime runtime = route(make_runtime(
        {0, 1}, QueryLabelOp::Or, {1, 1}, {max_u32, max_u32}, {0, 0}, max_u32));
    require(runtime.or_route_planner_profile.original_highfreq_cost_proxy ==
                std::numeric_limits<uint64_t>::max() &&
                runtime.or_route_planner_profile.emitted_highfreq_cost_proxy ==
                std::numeric_limits<uint64_t>::max(),
            "high-frequency proxy totals wrapped instead of saturating");
}

void test_and_routing_is_unchanged()
{
    const GraphRuntime fixture = make_runtime(
        {0, 1}, QueryLabelOp::And, {5, 100}, {100, 1}, {0, 0});
    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(fixture);
    configure_or_route_planner("cost_aware");
    const GraphRuntime cost_aware = route(fixture);
    configure_or_route_planner("coverage_cost");
    const GraphRuntime coverage_cost = route(fixture);
    configure_or_route_planner("relevance_coverage_cost");
    const GraphRuntime relevance_coverage_cost = route(fixture);
    require(route_tuples(cost_aware) == route_tuples(legacy),
            "OR planner mode changed AND routing metadata");
    require(route_tuples(coverage_cost) == route_tuples(legacy),
            "coverage_cost changed AND routing metadata");
    require(route_tuples(relevance_coverage_cost) == route_tuples(legacy),
            "relevance_coverage_cost changed AND routing metadata");
    require(cost_aware.query_filter_labels == std::vector<uint32_t>({0}),
            "AND routing no longer uses the legacy posting-size choice");
    require(cost_aware.or_route_planner_profile.queries == 0,
            "AND query was counted by the OR planner");
}

void test_pure_range_routing_is_unchanged()
{
    GraphRuntime fixture = make_runtime(
        {}, QueryLabelOp::And, {10}, {10}, {0});
    fixture.base_vector_metadata.rows = 10;
    fixture.original_query_ranges.range_count = 1;
    fixture.original_query_ranges.modes = {3};
    fixture.original_query_ranges.lower_bounds = {0.0f};
    fixture.original_query_ranges.upper_bounds = {1.0f};
    fixture.label_intervals.has_interval = {1};
    fixture.label_intervals.range_indices = {0};
    fixture.label_intervals.level_indices = {0};
    fixture.label_intervals.bin_indices = {0};
    fixture.label_intervals.lower_bounds = {0.0f};
    fixture.label_intervals.upper_bounds = {1.0f};

    configure_or_route_planner("legacy");
    const GraphRuntime legacy = route(fixture);
    configure_or_route_planner("coverage_cost");
    const GraphRuntime coverage_cost = route(fixture);
    configure_or_route_planner("relevance_coverage_cost");
    const GraphRuntime relevance_coverage_cost = route(fixture);
    const std::string model_path = temporary_config_path("_range_model.json");
    write_model_fixture(model_path, fixture, 4, 123);
    configure_marginal_fixture(fixture, model_path, 4, 123);
    const GraphRuntime marginal_accuracy_cost = route(fixture);
    require(route_tuples(coverage_cost) == route_tuples(legacy),
            "coverage_cost changed pure-range routing metadata");
    require(route_tuples(relevance_coverage_cost) == route_tuples(legacy),
            "relevance_coverage_cost changed pure-range routing metadata");
    require(route_tuples(marginal_accuracy_cost) == route_tuples(legacy),
            "marginal_accuracy_cost changed pure-range routing metadata");
    require(coverage_cost.or_route_planner_profile.queries == 0,
            "pure-range query was counted by the OR planner");
    require(marginal_accuracy_cost.or_route_planner_profile.sketch_labels == 0,
            "pure-range query caused active-label sketch loading");
    std::remove(model_path.c_str());
}

}  // namespace

int main()
{
    ScopedRouteCaps caps;
    test_hybrid_manifest_ten_subtype_dnf();
    test_hybrid_clause_local_route_materialization(caps);
    test_config_loading();
    test_default_and_route_set_preservation();
    test_lowfreq_exact_cost_ordering();
    test_duplicate_group_filter_and_residual_metadata_preservation();
    test_highfreq_proxy_ordering_and_ties();
    test_coverage_retention_then_cost_scheduling(caps);
    test_relevance_retention_then_cost_scheduling(caps);
    test_equal_relevance_uses_cardinality_before_cost(caps);
    test_categorical_or_defaults_to_exact_relevance(caps);
    test_manifest_categorical_cap_uses_coverage(caps);
    test_uncapped_relevance_mode_preserves_multiset();
    test_fuzzy_cap_policy_selects_label_identity();
    test_pre_dnf_fuzzy_cap();
    test_singleton_dnf_large_cap_and_lowfreq();
    test_compact_singleton_planner_equivalence(caps);
    test_misaligned_occurrence_metadata_is_rejected();
    test_invalid_occurrence_metadata_is_rejected_before_routing();
    test_group_union_is_not_original_label_coverage(caps);
    test_lowfreq_grouped_and_singleton_cardinality_is_cached_once();
    test_failed_cardinality_derivation_does_not_poison_retry();
    test_uncapped_coverage_cost_preserves_multiset();
    test_grouped_coverage_metadata_is_required();
    test_caps_apply_after_cost_ordering(caps);
    test_coverage_caps_skip_blocked_highfreq_and_fill_total(caps);
    test_legacy_alias_caps_and_cross_run_reset(caps);
    test_marginal_model_contract_rejections();
    test_marginal_model_size_and_diagnostics_gates();
    test_marginal_model_runtime_cap_binding(caps);
    test_or_route_calibration_trace_mapping_order_and_reentry();
    test_or_route_calibration_trace_cap_conflict(caps);
    test_or_route_calibration_trace_bypasses_and_and_range();
    test_marginal_overlap_beats_cardinality_only(caps);
    test_base_coverage_and_marginal_overlap_are_separate(caps);
    test_marginal_hard_relevance_guard(caps);
    test_fallback_utility_is_not_cheapest_first(caps);
    test_marginal_uncapped_small_k_determinism_and_reentry();
    test_marginal_sidecar_loader_contracts(caps);
    test_large_runtime_sketch_fallback_is_rejected();
    test_marginal_mode_does_not_change_and_routing();
    test_profile_cost_saturation();
    test_and_routing_is_unchanged();
    test_pure_range_routing_is_unchanged();
    configure_or_route_planner("legacy");
    std::puts("OR route planner tests passed");
    return 0;
}
