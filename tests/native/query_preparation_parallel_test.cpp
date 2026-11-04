// Reuse the existing semantic fixtures; exercise full input-order chunk merging.
#define main existing_or_route_test_main
#include "or_route_planner_test.cpp"
#undef main
#include "planning/query_prepare_parallel.h"
#include "index/range_posting_index.h"
#include "planning/route_predicate_plan.h"

// Disable the optimized selector explicitly when invoking the eager oracle.
// query_prepare_optimized controls representation, not this existing selector.
class ScopedCompactPlanner {
    std::optional<std::string> previous;
public:
    explicit ScopedCompactPlanner(bool enabled) {
        if (const char* value = std::getenv("GO_FANN_COMPACT_CLAUSE_PLANNER")) previous = value;
        setenv("GO_FANN_COMPACT_CLAUSE_PLANNER", enabled ? "1" : "0", 1);
    }
    ~ScopedCompactPlanner() {
        if (previous) setenv("GO_FANN_COMPACT_CLAUSE_PLANNER", previous->c_str(), 1);
        else unsetenv("GO_FANN_COMPACT_CLAUSE_PLANNER");
    }
};

static GraphRuntime repeated_fixture(uint32_t count, bool fp32,
                                      const HybridManifestParseTestResult* manifest = nullptr) {
    auto single = make_fuzzy_manifest_cap_fixture(true);
    if (manifest) {
        single.original_query_clauses = manifest->clauses;
        single.original_query_labels_matrix = manifest->union_labels;
        single.original_query_label_ops = {manifest->top_op};
    }
    auto result = single;
    result.original_query_count = count;
    result.original_query_vectors.rows = count;
    result.original_query_vectors.data.clear();
    result.original_query_label_ops.assign(count, single.original_query_label_ops[0]);
    auto& labels = result.original_query_labels_matrix;
    labels.rows = count; labels.indptr.assign(1, 0); labels.indices.clear();
    labels.row_labels.clear(); labels.source_tiers.clear(); labels.relevance_ranks.clear();
    auto& clauses = result.original_query_clauses;
    clauses.rows = count; clauses.query_indptr.assign(1, 0); clauses.label_indptr.assign(1, 0);
    clauses.labels.clear(); clauses.source_tiers.clear(); clauses.relevance_ranks.clear();
    clauses.expressions.clear(); clauses.atom_labels.clear(); clauses.atom_source_tiers.clear(); clauses.atom_relevance_ranks.clear();
    clauses.top_branch_ids.clear(); clauses.uses_original_range.clear();
    clauses.logical_clause_count = 0; clauses.descriptor_indptr.clear();
    if (clauses.virtual_descriptors) clauses.descriptor_indptr.push_back(0);
    clauses.branch_caps.clear(); clauses.branch_cap_is_fuzzy.clear();
    result.original_query_ranges.rows = count;
    result.original_query_ranges.modes.clear(); result.original_query_ranges.lower_bounds.clear(); result.original_query_ranges.upper_bounds.clear();
    const auto append = [](auto& a, const auto& b) { a.insert(a.end(), b.begin(), b.end()); };
    for (uint32_t q = 0; q < count; ++q) {
        append(result.original_query_vectors.data, single.original_query_vectors.data);
        append(labels.indices, single.original_query_labels_matrix.indices);
        append(labels.row_labels, single.original_query_labels_matrix.row_labels);
        append(labels.source_tiers, single.original_query_labels_matrix.source_tiers);
        append(labels.relevance_ranks, single.original_query_labels_matrix.relevance_ranks);
        labels.indptr.push_back(labels.indices.size());
        const auto base = clauses.labels.size();
        const auto& original = single.original_query_clauses;
        if (!original.expressions.empty()) {
            auto expression = original.expressions[0];
            for (auto& atom : expression.atoms) atom.begin += clauses.atom_labels.size();
            clauses.expressions.push_back(expression);
        }
        append(clauses.atom_labels, original.atom_labels);
        append(clauses.atom_source_tiers, original.atom_source_tiers);
        append(clauses.atom_relevance_ranks, original.atom_relevance_ranks);
        append(clauses.labels, original.labels); append(clauses.source_tiers, original.source_tiers);
        append(clauses.relevance_ranks, original.relevance_ranks);
        append(clauses.top_branch_ids, original.top_branch_ids); append(clauses.uses_original_range, original.uses_original_range);
        for (size_t i = 1; i < original.label_indptr.size(); ++i) clauses.label_indptr.push_back(base + original.label_indptr[i]);
        if (clauses.virtual_descriptors) {
            clauses.logical_clause_count += original.clause_count();
            clauses.descriptor_indptr.push_back(clauses.top_branch_ids.size());
        }
        clauses.query_indptr.push_back(clauses.clause_count());
        clauses.branch_caps.push_back(q % 4 == 0 ? 0 : q % 4 == 1 ? 1 : q % 4 == 2 ? 16 : 64);
        clauses.branch_cap_is_fuzzy.push_back(1);
        append(result.original_query_ranges.modes, single.original_query_ranges.modes);
        append(result.original_query_ranges.lower_bounds, single.original_query_ranges.lower_bounds);
        append(result.original_query_ranges.upper_bounds, single.original_query_ranges.upper_bounds);
    }
    if (fp32) {
        result.vector_scalar_type = DenseVectorScalarType::Float32;
        result.original_query_float_vectors.rows = count;
        result.original_query_float_vectors.cols = result.original_query_vectors.cols;
        for (uint8_t v : result.original_query_vectors.data) result.original_query_float_vectors.data.push_back(float(v) / 16);
        result.original_query_vectors.data.clear();
    }
    return result;
}

static void compare(const GraphRuntime& a, const GraphRuntime& b) {
    if (b.shared_route_query_vectors) {
        require(b.query_vectors.data.empty() && b.query_float_vectors.data.empty(), "shared routes replicated vectors");
        for (uint32_t route = 0; route < a.query_vectors.rows; ++route) {
            if (a.vector_scalar_type == DenseVectorScalarType::Uint8)
                require(std::equal(a.route_query_u8(route), a.route_query_u8(route) + a.query_vectors.cols,
                                   b.route_query_u8(route)), "shared uint8 route vector differs");
            else require(std::equal(a.route_query_float(route), a.route_query_float(route) + a.query_vectors.cols,
                                    b.route_query_float(route)), "shared FP32 route vector differs");
        }
    } else {
        require(a.query_vectors.data == b.query_vectors.data, "parallel route mismatch: query_vectors.data");
        require(a.query_float_vectors.data == b.query_float_vectors.data, "parallel route mismatch: query_float_vectors.data");
    }
    require(a.query_vectors.rows == b.query_vectors.rows, "parallel route mismatch: query_vectors.rows");
    require(a.query_float_vectors.rows == b.query_float_vectors.rows, "parallel route mismatch: query_float_vectors.rows");
    require(a.query_labels_matrix.indices == b.query_labels_matrix.indices, "parallel route mismatch: query_labels_matrix.indices");
    require(a.query_labels_matrix.indptr == b.query_labels_matrix.indptr, "parallel route mismatch: query_labels_matrix.indptr");
    require(a.query_labels_matrix.row_labels == b.query_labels_matrix.row_labels, "parallel route mismatch: query_labels_matrix.row_labels");
    require(a.query_labels_matrix.source_tiers == b.query_labels_matrix.source_tiers, "parallel route mismatch: query_labels_matrix.source_tiers");
    require(a.query_labels_matrix.relevance_ranks == b.query_labels_matrix.relevance_ranks, "parallel route mismatch: query_labels_matrix.relevance_ranks");
    require(a.high_query_indices == b.high_query_indices, "parallel route mismatch: high_query_indices");
    require(a.low_query_indices == b.low_query_indices, "parallel route mismatch: low_query_indices");
    require(a.query_group_ids == b.query_group_ids, "parallel route mismatch: query_group_ids");
    require(a.query_filter_labels == b.query_filter_labels, "parallel route mismatch: query_filter_labels");
    require(a.query_filter_bits == b.query_filter_bits, "parallel route mismatch: query_filter_bits");
    require(a.start_local_ids == b.start_local_ids, "parallel route mismatch: start_local_ids");
    require(a.query_is_lowfreq == b.query_is_lowfreq, "parallel route mismatch: query_is_lowfreq");
    require(a.route_to_original_query == b.route_to_original_query, "parallel route mismatch: route_to_original_query");
    require(a.route_to_clause == b.route_to_clause, "parallel route mismatch: route_to_clause");
    require(a.route_clause_indptr == b.route_clause_indptr, "parallel route mismatch: route_clause_indptr");
    require(a.route_clause_ids == b.route_clause_ids, "parallel route mismatch: route_clause_ids");
    require(a.route_top_branch_ids == b.route_top_branch_ids, "parallel route mismatch: route_top_branch_ids");
    require(a.route_enabled == b.route_enabled, "parallel route mismatch: route_enabled");
    require(a.route_providers == b.route_providers, "parallel route mismatch: route_providers");
    require(a.route_execution_modes == b.route_execution_modes, "parallel route mismatch: route_execution_modes");
    require(a.route_candidate_sources == b.route_candidate_sources, "parallel route mismatch: route_candidate_sources");
    require(a.query_ranges.modes == b.query_ranges.modes, "parallel route mismatch: query_ranges.modes");
    require(a.query_ranges.lower_bounds == b.query_ranges.lower_bounds, "parallel route mismatch: query_ranges.lower_bounds");
    require(a.query_ranges.upper_bounds == b.query_ranges.upper_bounds, "parallel route mismatch: query_ranges.upper_bounds");
    require(a.or_route_planner_profile.mode == b.or_route_planner_profile.mode, "parallel route mismatch: or_route_planner_profile.mode");
    require(a.or_route_planner_profile.total_cap == b.or_route_planner_profile.total_cap, "parallel route mismatch: or_route_planner_profile.total_cap");
    require(a.or_route_planner_profile.highfreq_cap == b.or_route_planner_profile.highfreq_cap, "parallel route mismatch: or_route_planner_profile.highfreq_cap");
    require(a.or_route_planner_profile.queries == b.or_route_planner_profile.queries, "parallel route mismatch: or_route_planner_profile.queries");
    require(a.or_route_planner_profile.original_routes == b.or_route_planner_profile.original_routes, "parallel route mismatch: or_route_planner_profile.original_routes");
    require(a.or_route_planner_profile.emitted_routes == b.or_route_planner_profile.emitted_routes, "parallel route mismatch: or_route_planner_profile.emitted_routes");
    require(a.or_route_planner_profile.original_highfreq_routes == b.or_route_planner_profile.original_highfreq_routes, "parallel route mismatch: or_route_planner_profile.original_highfreq_routes");
    require(a.or_route_planner_profile.original_lowfreq_routes == b.or_route_planner_profile.original_lowfreq_routes, "parallel route mismatch: or_route_planner_profile.original_lowfreq_routes");
    require(a.or_route_planner_profile.emitted_highfreq_routes == b.or_route_planner_profile.emitted_highfreq_routes, "parallel route mismatch: or_route_planner_profile.emitted_highfreq_routes");
    require(a.or_route_planner_profile.emitted_lowfreq_routes == b.or_route_planner_profile.emitted_lowfreq_routes, "parallel route mismatch: or_route_planner_profile.emitted_lowfreq_routes");
    require(a.or_route_planner_profile.original_exact_source_routes == b.or_route_planner_profile.original_exact_source_routes, "parallel route mismatch: or_route_planner_profile.original_exact_source_routes");
    require(a.or_route_planner_profile.emitted_exact_source_routes == b.or_route_planner_profile.emitted_exact_source_routes, "parallel route mismatch: or_route_planner_profile.emitted_exact_source_routes");
    require(a.or_route_planner_profile.original_direct_source_routes == b.or_route_planner_profile.original_direct_source_routes, "parallel route mismatch: or_route_planner_profile.original_direct_source_routes");
    require(a.or_route_planner_profile.emitted_direct_source_routes == b.or_route_planner_profile.emitted_direct_source_routes, "parallel route mismatch: or_route_planner_profile.emitted_direct_source_routes");
    require(a.or_route_planner_profile.original_fallback_source_routes == b.or_route_planner_profile.original_fallback_source_routes, "parallel route mismatch: or_route_planner_profile.original_fallback_source_routes");
    require(a.or_route_planner_profile.emitted_fallback_source_routes == b.or_route_planner_profile.emitted_fallback_source_routes, "parallel route mismatch: or_route_planner_profile.emitted_fallback_source_routes");
    require(a.or_route_planner_profile.retained_relevance_rank_sum == b.or_route_planner_profile.retained_relevance_rank_sum, "parallel route mismatch: or_route_planner_profile.retained_relevance_rank_sum");
    require(a.or_route_planner_profile.retained_relevance_rank_max == b.or_route_planner_profile.retained_relevance_rank_max, "parallel route mismatch: or_route_planner_profile.retained_relevance_rank_max");
    require(a.or_route_planner_profile.reordered_queries == b.or_route_planner_profile.reordered_queries, "parallel route mismatch: or_route_planner_profile.reordered_queries");
    require(a.or_route_planner_profile.reordered_routes == b.or_route_planner_profile.reordered_routes, "parallel route mismatch: or_route_planner_profile.reordered_routes");
    require(a.or_route_planner_profile.capped_queries == b.or_route_planner_profile.capped_queries, "parallel route mismatch: or_route_planner_profile.capped_queries");
    require(a.or_route_planner_profile.capped_routes == b.or_route_planner_profile.capped_routes, "parallel route mismatch: or_route_planner_profile.capped_routes");
    require(a.or_route_planner_profile.total_cap_capped_routes == b.or_route_planner_profile.total_cap_capped_routes, "parallel route mismatch: or_route_planner_profile.total_cap_capped_routes");
    require(a.or_route_planner_profile.highfreq_cap_capped_routes == b.or_route_planner_profile.highfreq_cap_capped_routes, "parallel route mismatch: or_route_planner_profile.highfreq_cap_capped_routes");
    require(a.or_route_planner_profile.manifest_cap_queries == b.or_route_planner_profile.manifest_cap_queries, "parallel route mismatch: or_route_planner_profile.manifest_cap_queries");
    require(a.or_route_planner_profile.manifest_cap_max == b.or_route_planner_profile.manifest_cap_max, "parallel route mismatch: or_route_planner_profile.manifest_cap_max");
    require(a.or_route_planner_profile.original_exact_lowfreq_rows == b.or_route_planner_profile.original_exact_lowfreq_rows, "parallel route mismatch: or_route_planner_profile.original_exact_lowfreq_rows");
    require(a.or_route_planner_profile.emitted_exact_lowfreq_rows == b.or_route_planner_profile.emitted_exact_lowfreq_rows, "parallel route mismatch: or_route_planner_profile.emitted_exact_lowfreq_rows");
    require(a.or_route_planner_profile.original_highfreq_cost_proxy == b.or_route_planner_profile.original_highfreq_cost_proxy, "parallel route mismatch: or_route_planner_profile.original_highfreq_cost_proxy");
    require(a.or_route_planner_profile.emitted_highfreq_cost_proxy == b.or_route_planner_profile.emitted_highfreq_cost_proxy, "parallel route mismatch: or_route_planner_profile.emitted_highfreq_cost_proxy");
    require(a.or_route_planner_profile.original_exact_label_cardinality_sum == b.or_route_planner_profile.original_exact_label_cardinality_sum, "parallel route mismatch: or_route_planner_profile.original_exact_label_cardinality_sum");
    require(a.or_route_planner_profile.emitted_exact_label_cardinality_sum == b.or_route_planner_profile.emitted_exact_label_cardinality_sum, "parallel route mismatch: or_route_planner_profile.emitted_exact_label_cardinality_sum");
    require(a.or_route_planner_profile.sketch_labels == b.or_route_planner_profile.sketch_labels, "parallel route mismatch: or_route_planner_profile.sketch_labels");
    require(a.or_route_planner_profile.sketch_hashes == b.or_route_planner_profile.sketch_hashes, "parallel route mismatch: or_route_planner_profile.sketch_hashes");
    require(a.or_route_planner_profile.sketch_bytes == b.or_route_planner_profile.sketch_bytes, "parallel route mismatch: or_route_planner_profile.sketch_bytes");
    require(a.or_route_planner_profile.sketch_sidecar_bytes_read == b.or_route_planner_profile.sketch_sidecar_bytes_read, "parallel route mismatch: or_route_planner_profile.sketch_sidecar_bytes_read");
    require(a.or_route_planner_profile.sketch_build_rows == b.or_route_planner_profile.sketch_build_rows, "parallel route mismatch: or_route_planner_profile.sketch_build_rows");
    require(a.or_route_planner_profile.retained_zero_marginal_routes == b.or_route_planner_profile.retained_zero_marginal_routes, "parallel route mismatch: or_route_planner_profile.retained_zero_marginal_routes");
    require(a.or_route_planner_profile.exact_cardinality_used == b.or_route_planner_profile.exact_cardinality_used, "parallel route mismatch: or_route_planner_profile.exact_cardinality_used");
    require(a.or_route_planner_profile.estimated_union_coverage_sum == b.or_route_planner_profile.estimated_union_coverage_sum, "parallel route mismatch: or_route_planner_profile.estimated_union_coverage_sum");
    require(a.or_route_planner_profile.retained_estimated_marginal_gain_sum == b.or_route_planner_profile.retained_estimated_marginal_gain_sum, "parallel route mismatch: or_route_planner_profile.retained_estimated_marginal_gain_sum");
    require(a.or_route_planner_profile.original_predicted_latency_us_sum == b.or_route_planner_profile.original_predicted_latency_us_sum, "parallel route mismatch: or_route_planner_profile.original_predicted_latency_us_sum");
    require(a.or_route_planner_profile.emitted_predicted_latency_us_sum == b.or_route_planner_profile.emitted_predicted_latency_us_sum, "parallel route mismatch: or_route_planner_profile.emitted_predicted_latency_us_sum");
    require(a.or_route_planner_profile.emitted_quality_sum == b.or_route_planner_profile.emitted_quality_sum, "parallel route mismatch: or_route_planner_profile.emitted_quality_sum");
    require(a.or_route_planner_profile.emitted_utility_sum == b.or_route_planner_profile.emitted_utility_sum, "parallel route mismatch: or_route_planner_profile.emitted_utility_sum");
    require(a.or_route_planner_profile.calibration_id == b.or_route_planner_profile.calibration_id, "parallel route mismatch: or_route_planner_profile.calibration_id");
}

uint64_t align_up(uint64_t value)
{
    constexpr uint64_t alignment = 64;
    return ((value + alignment - 1) / alignment) * alignment;
}

void write_zeroes(std::ofstream& output, uint64_t target)
{
    const uint64_t current = static_cast<uint64_t>(output.tellp());
    require(current <= target, "range posting fixture offsets overlap");
    const std::vector<char> zeroes(static_cast<size_t>(target - current), 0);
    output.write(zeroes.data(), static_cast<std::streamsize>(zeroes.size()));
}

void write_index(const std::filesystem::path& path)
{
    const std::vector<float> values{15.0f, 25.0f};
    const std::vector<uint64_t> offsets{0, 4, 8};
    const std::vector<uint32_t> ids{1, 3, 5, 7, 0, 2, 4, 6};
    go_fann::range_posting::FileHeader header{};
    std::memcpy(header.magic, go_fann::range_posting::kMagic, sizeof(go_fann::range_posting::kMagic));
    header.version = go_fann::range_posting::kVersion;
    header.header_bytes = sizeof(go_fann::range_posting::FileHeader) + sizeof(go_fann::range_posting::DimensionDescriptor);
    header.row_count = ids.size();
    header.range_count = 1;

    go_fann::range_posting::DimensionDescriptor descriptor{};
    descriptor.range_index = 0;
    descriptor.unique_count = values.size();
    descriptor.row_count = ids.size();
    uint64_t cursor = align_up(header.header_bytes);
    descriptor.values_offset = cursor;
    descriptor.values_bytes = values.size() * sizeof(float);
    cursor = align_up(cursor + descriptor.values_bytes);
    descriptor.offsets_offset = cursor;
    descriptor.offsets_bytes = offsets.size() * sizeof(uint64_t);
    cursor = align_up(cursor + descriptor.offsets_bytes);
    descriptor.sample_ids_offset = cursor;
    descriptor.sample_ids_bytes = ids.size() * sizeof(uint32_t);
    header.file_bytes = align_up(cursor + descriptor.sample_ids_bytes);

    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(reinterpret_cast<const char*>(&descriptor), sizeof(descriptor));
    write_zeroes(output, descriptor.values_offset);
    output.write(reinterpret_cast<const char*>(values.data()), descriptor.values_bytes);
    write_zeroes(output, descriptor.offsets_offset);
    output.write(reinterpret_cast<const char*>(offsets.data()), descriptor.offsets_bytes);
    write_zeroes(output, descriptor.sample_ids_offset);
    output.write(reinterpret_cast<const char*>(ids.data()), descriptor.sample_ids_bytes);
    write_zeroes(output, header.file_bytes);
    require(output.good(), "unable to write range posting fixture");
}

static void test_deferred_range_provider_parity() {
    const std::filesystem::path path = "/tmp/deferred-range-" + std::to_string(getpid()) + ".bin";
    write_index(path);
    const auto index = go_fann::range_posting::Index::load(path.string());
    std::filesystem::remove(path);
    for (const std::string op : {"and", "or"}) for (const std::string filter : {"auto", "pre", "post"})
    for (const std::string policy : {"auto", "label_prefer", "range_prefer", "dual"})
    for (uint64_t threshold : {3U, 8U})
    for (uint32_t cap : {0U, 1U, 2U, 16U, 64U})
    for (const std::string cap_policy : {"legacy", "cost_aware", "coverage_cost", "relevance_coverage_cost"}) {
        OrRoutePlannerConfig planner;
        planner.order_mode = cap_policy; planner.fuzzy_cap_policy = cap_policy;
        configure_or_route_planner(planner);
        configure_query_preparation({1, true});
        const auto row = hybrid_manifest_row(hybrid_fuzzy_atom(), hybrid_range_atom(), op);
        const auto eager = evaluate_hybrid_query_manifest_row_for_test(row, {0, 1, 2}, {1, 1, 1}, {2, 1, 0}, 3);
        const auto lazy = evaluate_hybrid_query_manifest_row_for_test(row, {0, 1, 2}, {1, 1, 1}, {2, 1, 0}, 3, true);
        const auto fixture = [&](const auto& source) {
            auto runtime = repeated_fixture(1, false, &source);
            runtime.original_query_clauses.branch_caps = {cap};
            runtime.supported_query_route_providers |= kQueryRouteProviderRangePosting;
            runtime.base_vectors.rows = 8;
            runtime.base_range_count = 1;
            runtime.original_query_ranges.rows = 1;
            runtime.original_query_ranges.range_count = 1;
            runtime.original_query_ranges.modes = {3};
            runtime.original_query_ranges.lower_bounds = {10};
            runtime.original_query_ranges.upper_bounds = {20};
            runtime.data_labels.rows = 8; runtime.data_labels.cols = 3;
            runtime.data_labels.indptr = {0};
            for (uint32_t mask = 0; mask < 8; ++mask) {
                for (uint32_t label = 0; label < 3; ++label)
                    if (mask & (1U << label)) runtime.data_labels.indices.push_back(label);
                runtime.data_labels.indptr.push_back(runtime.data_labels.indices.size());
                runtime.base_range_values.push_back(mask % 2 ? 15.0f : 25.0f);
            }
            return runtime;
        };
        auto reference = fixture(eager), actual = fixture(lazy);
        RangeAccessPlannerConfig config;
        config.mode = "select"; config.and_filter_mode = filter;
        config.range_label_and_policy = policy; config.scan_threshold = threshold;
        config.range_graph_hybrid_and_enabled = config.range_graph_hybrid_or_enabled = true;
        configure_query_preparation({1, false});
        {
            ScopedCompactPlanner oracle(false);
            validate_and_prepare_queries(reference, 2, "greedy_min_posting");
        }
        plan_graph_range_access(reference, index, config);
        coalesce_graph_routes(reference);
        configure_query_preparation({8, true});
        {
            ScopedCompactPlanner optimized(true);
            validate_and_prepare_queries(actual, 2, "greedy_min_posting", {}, true);
        }
        plan_graph_range_access(actual, index, config);
        require(actual.query_vectors.data.empty() && actual.query_ranges.modes.empty() &&
                actual.query_labels_matrix.indices.empty(), "provider planner materialized execution buffers");
        coalesce_graph_routes(actual);
        materialize_graph_route_payloads(actual);
        compare(reference, actual);
#define RANGE_EQUAL(member) require(reference.member == actual.member, "deferred range mismatch: " #member)
        RANGE_EQUAL(route_range_posting_begins); RANGE_EQUAL(route_range_posting_sizes); RANGE_EQUAL(route_range_posting_indices);
        RANGE_EQUAL(range_graph_and_original_queries); RANGE_EQUAL(range_graph_or_original_queries);
        RANGE_EQUAL(range_graph_clause_indptr); RANGE_EQUAL(range_graph_clause_ids);
#undef RANGE_EQUAL
        (void)go_fann::route_predicate::compile(actual, go_fann::route_predicate::Producer::Graph);
        for (uint32_t sample = 0; sample < 8; ++sample) {
            bool original_match = false;
            for (uint32_t clause = 0; clause < actual.original_query_clauses.clause_count(); ++clause) {
                const auto labels = actual.original_query_clauses.clause_labels(clause);
                bool matches = true;
                for (auto label = labels.begin(); label != labels.end(); ++label) matches &= (sample & (1U << *label)) != 0;
                if (actual.original_query_clauses.clause_uses_range(clause)) matches &= sample % 2 != 0;
                original_match |= matches;
            }
            const bool expected = op == "and" ? (sample != 0 && sample % 2) : (sample != 0 || sample % 2);
            require(original_match == expected, "lazy original predicate differs from bitmask oracle");
            for (uint32_t route = 0; route < actual.route_to_original_query.size(); ++route)
                require(sample_matches_route_predicate(actual, route, sample) ==
                        sample_matches_route_predicate(reference, route, sample), "provider residual changed exact validity");
        }
        GraphQueryState saved;
        swap_graph_query_state(actual, saved); swap_graph_query_state(actual, saved);
        compare(reference, actual);
    }
}

static void test_lazy_clause_selection() {
    const auto categorical = hybrid_label_atom("categorical", "single", 0);
    const auto exact = hybrid_label_atom("string", "exact", 1);
    const auto fuzzy = hybrid_fuzzy_atom();
    const auto range = hybrid_range_atom();
    const std::vector<std::pair<nlohmann::json, nlohmann::json>> pairs{
        {categorical, exact}, {categorical, fuzzy}, {categorical, range}, {exact, range}, {fuzzy, range}};
    for (auto atoms : pairs) for (bool reverse : {false, true}) for (const std::string op : {"and", "or"}) {
        if (reverse) std::swap(atoms.first, atoms.second);
        const auto row = hybrid_manifest_row(atoms.first, atoms.second, op);
        configure_query_preparation({1, true});
        const bool fuzzy_row = atoms.first.value("subtype", "") == "fuzzy" || atoms.second.value("subtype", "") == "fuzzy";
        const std::vector<uint32_t> labels = fuzzy_row ? std::vector<uint32_t>{0, 1, 2} : std::vector<uint32_t>{};
        const std::vector<uint8_t> tiers = fuzzy_row ? std::vector<uint8_t>{1, 1, 1} : std::vector<uint8_t>{};
        const std::vector<uint32_t> ranks = fuzzy_row ? std::vector<uint32_t>{7, 2, 1} : std::vector<uint32_t>{};
        const auto eager = evaluate_hybrid_query_manifest_row_for_test(row, labels, tiers, ranks, 3);
        const auto lazy = evaluate_hybrid_query_manifest_row_for_test(row, labels, tiers, ranks, 3, true);
        require(lazy.clauses.labels.empty() && lazy.clauses.source_tiers.empty() &&
                lazy.clauses.relevance_ranks.empty(), "lazy original DNF expanded clause occurrences");
        require(eager.clauses.query_indptr == lazy.clauses.query_indptr,
                "lazy source changed logical clause order/identity");
        require(lazy.clauses.virtual_descriptors && lazy.clauses.label_indptr.size() == 1 &&
                lazy.clauses.top_branch_ids.empty() && lazy.clauses.uses_original_range.empty() &&
                lazy.clauses.descriptor_shape_valid(), "virtual DNF allocated expanded descriptors");
        for (uint32_t clause = 0; clause < eager.clauses.clause_count(); ++clause) {
            require(eager.clauses.clause_branch(clause) == lazy.clauses.clause_branch(clause) &&
                    eager.clauses.clause_uses_range(clause) == lazy.clauses.clause_uses_range(clause),
                    "virtual clause changed branch/range identity");
            const auto a = eager.clauses.clause_labels(clause), b = lazy.clauses.clause_labels(clause);
            require(a.valid && b.valid && a.count == b.count, "lazy clause shape differs from eager oracle");
            for (size_t i = 0; i < a.count; ++i)
                require(a.begin()[i] == b.begin()[i] && a.tier(i) == b.tier(i) && a.rank(i) == b.rank(i),
                        "lazy clause changed duplicate-label occurrence resolution");
        }
        bool has_range = false;
        for (size_t clause = 0; clause < lazy.clauses.clause_count(); ++clause)
            has_range |= lazy.clauses.clause_uses_range(clause);
        if (has_range) continue;
        for (const std::string policy : {"legacy", "cost_aware", "coverage_cost", "relevance_coverage_cost"}) {
            OrRoutePlannerConfig planner; planner.order_mode = policy; planner.fuzzy_cap_policy = policy;
            configure_or_route_planner(planner);
            for (bool fp32 : {false, true}) {
                auto reference = repeated_fixture(257, fp32, &eager);
                // Non-fuzzy shapes do not have a fuzzy cap.
                if (atoms.first.value("subtype", "") != "fuzzy" && atoms.second.value("subtype", "") != "fuzzy")
                    reference.original_query_clauses.branch_caps.assign(257, 0);
                configure_query_preparation({1, false});
                {
                    ScopedCompactPlanner oracle(false);
                    validate_and_prepare_queries(reference, 2, "per_label_multi_route");
                }
                coalesce_graph_routes(reference);
                for (uint32_t workers : {1U, 4U, 8U}) {
                    auto actual = repeated_fixture(257, fp32, &lazy);
                    actual.shared_route_query_vectors = true;
                    if (atoms.first.value("subtype", "") != "fuzzy" && atoms.second.value("subtype", "") != "fuzzy")
                        actual.original_query_clauses.branch_caps.assign(257, 0);
                    configure_query_preparation({workers, true});
                    {
                        ScopedCompactPlanner optimized(true);
                        validate_and_prepare_queries(actual, 2, "per_label_multi_route", {}, true);
                    }
                    require(actual.query_vectors.data.empty() && actual.query_labels_matrix.indices.empty(),
                            "lazy source caused eager execution allocation");
                    coalesce_graph_routes(actual);
                    materialize_graph_route_payloads(actual);
                    compare(reference, actual);
                }
            }
        }
        auto invalid = lazy.clauses;
        invalid.expressions[0].atoms[0].begin = UINT64_MAX;
        require(!invalid.clause_labels(0).valid, "invalid lazy atom reference accepted");
    }
}

int main() {
    test_deferred_range_provider_parity();
    test_lazy_clause_selection();
    ScopedRouteCaps caps;
    for (const std::string policy : {"legacy", "cost_aware", "coverage_cost", "relevance_coverage_cost"}) {
        OrRoutePlannerConfig planner; planner.order_mode = policy; planner.fuzzy_cap_policy = policy;
        configure_or_route_planner(planner);
        for (bool fp32 : {false, true}) {
            const auto fixture = repeated_fixture(257, fp32);
            std::vector<uint8_t> mask(257, 1);
            for (size_t i = 0; i < mask.size(); i += 17) mask[i] = 0;
            auto reference = fixture;
            configure_query_preparation({1, false});
            validate_and_prepare_queries(reference, 2, "greedy_min_posting", mask);
            coalesce_graph_routes(reference);
            for (uint32_t threads : {1U, 4U, 8U}) {
                auto actual = fixture;
                actual.shared_route_query_vectors = true;
                configure_query_preparation({threads, true});
                validate_and_prepare_queries(actual, 2, "greedy_min_posting", mask, true);
                require(actual.query_vectors.data.empty() && actual.query_float_vectors.data.empty() &&
                        actual.query_labels_matrix.indices.empty() && actual.query_labels_matrix.row_labels.empty() &&
                        actual.query_ranges.modes.empty(), "selected route descriptors eagerly allocated payloads");
                coalesce_graph_routes(actual);
                require(actual.query_vectors.data.empty() && actual.query_labels_matrix.row_labels.empty() &&
                        actual.query_ranges.modes.empty(), "route union allocated execution payloads");
                materialize_graph_route_payloads(actual);
                compare(reference, actual);
                materialize_graph_route_payloads(actual);
                compare(reference, actual);
            }
        }
    }
    {
        auto invalid = repeated_fixture(257, false);
        configure_query_preparation({8, true});
        require_throws_contains([&] { validate_and_prepare_queries(invalid, 3, "greedy_min_posting"); },
                                "dimension mismatch", "parallel planner skipped parent validation");
    }
    // Identical token IDs with different occurrence metadata exercise exact
    // sibling/fuzzy duplicate resolution in flat DNF materialization.
    for (const std::string op : {"and", "or"}) {
        const auto row = hybrid_manifest_row(hybrid_label_atom("categorical", "single", 2), hybrid_fuzzy_atom(), op);
        configure_query_preparation({1, false});
        const auto old = evaluate_hybrid_query_manifest_row_for_test(row, {2, 3}, {tier(QueryLabelSourceTier::Direct), tier(QueryLabelSourceTier::Direct)}, {7, 1}, 8);
        configure_query_preparation({4, true});
        const auto fast = evaluate_hybrid_query_manifest_row_for_test(row, {2, 3}, {tier(QueryLabelSourceTier::Direct), tier(QueryLabelSourceTier::Direct)}, {7, 1}, 8);
        require(old.union_labels.indptr == fast.union_labels.indptr, "flat manifest mismatch: labels.indptr");
        require(old.union_labels.indices == fast.union_labels.indices, "flat manifest mismatch: labels.indices");
        require(old.union_labels.row_labels == fast.union_labels.row_labels, "flat manifest mismatch: labels.row_labels");
        require(old.union_labels.source_tiers == fast.union_labels.source_tiers, "flat manifest mismatch: labels.source_tiers");
        require(old.union_labels.relevance_ranks == fast.union_labels.relevance_ranks, "flat manifest mismatch: labels.relevance_ranks");
        require(old.clauses.query_indptr == fast.clauses.query_indptr, "flat manifest mismatch: clauses.query_indptr");
        require(old.clauses.branch_caps == fast.clauses.branch_caps, "flat manifest mismatch: clauses.branch_caps");
        require(old.clauses.branch_cap_is_fuzzy == fast.clauses.branch_cap_is_fuzzy, "flat manifest mismatch: clauses.branch_cap_is_fuzzy");
        require(old.clauses.label_indptr == fast.clauses.label_indptr, "flat manifest mismatch: clauses.label_indptr");
        require(old.clauses.labels == fast.clauses.labels, "flat manifest mismatch: clauses.labels");
        require(old.clauses.source_tiers == fast.clauses.source_tiers, "flat manifest mismatch: clauses.source_tiers");
        require(old.clauses.relevance_ranks == fast.clauses.relevance_ranks, "flat manifest mismatch: clauses.relevance_ranks");
        require(old.clauses.top_branch_ids == fast.clauses.top_branch_ids, "flat manifest mismatch: clauses.top_branch_ids");
        require(old.clauses.uses_original_range == fast.clauses.uses_original_range, "flat manifest mismatch: clauses.uses_original_range");
    }
    const std::string config_path = "/tmp/query-prepare-config-" + std::to_string(getpid()) + ".json";
    { std::ofstream out(config_path); out << R"({"query_prepare_threads":1,"query_prepare_optimized":false,"runtime":{"query_prepare_threads":16,"query_prepare_optimized":true}})"; }
    const auto parsed = go_fann::load_config_file(config_path);
    std::remove(config_path.c_str());
    require(parsed.query_prepare_threads == 16 && parsed.query_prepare_optimized == true, "query preparation nested config precedence");
    require_throws_contains([] { configure_query_preparation({257, true}); }, "[0, 256]", "invalid worker count accepted");
    require_throws_contains([] { go_fann::query_prepare::parallel_tasks(8, 4, [](size_t i) { if (i == 2 || i == 5) throw std::runtime_error(std::to_string(i)); }); }, "2", "parallel error order changed");
    configure_query_preparation({});
    std::puts("Online preparation parallel/serial equivalence passed");
    return 0;
}
