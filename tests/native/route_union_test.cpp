// Use the planner's existing fixture construction, not a duplicate planner.
#define main existing_or_route_test_main
#include "or_route_planner_test.cpp"
#undef main
#include "planning/route_predicate_plan.h"
#include "execution/async_io.h"

static GraphRuntime union_fixture(bool posting = false, bool grouped = false) {
    auto r = make_runtime({0,1,2}, QueryLabelOp::Or, {4,8,8},
                          posting ? std::vector<uint32_t>{0,0,0} : std::vector<uint32_t>{4,8,8},
                          posting ? std::vector<uint32_t>{4,8,8} : std::vector<uint32_t>{0,0,0});
    configure_hybrid_clause_fixture(r, {0,2}, {0,2,4}, {0,1,0,2}, {0,1}, {0,0});
    r.base_range_count = 1;
    r.base_vectors.rows = 8;
    r.data_labels.rows = 8; r.data_labels.cols = 3;
    r.data_labels.indptr = {0};
    for (uint32_t mask = 0; mask < 8; ++mask) {
        std::vector<uint32_t> labels;
        for (uint32_t label = 0; label < 3; ++label) if (mask & (1U << label)) labels.push_back(label);
        r.data_labels.indices.insert(r.data_labels.indices.end(), labels.begin(), labels.end());
        r.data_labels.indptr.push_back(r.data_labels.indices.size());
        r.data_labels.row_labels.push_back(labels);
        r.base_range_values.push_back(mask % 2 ? 15.0f : 25.0f);
    }
    if (grouped) {
        r.group_layout.label_to_group = {0,0,2};
        r.group_layout.groups = {{0,1},{},{2}};
        r.metadata.label_filter_bit = {1,2,1};
        r.seed_by_original_label = {0,1,0};
        r.seed_selection_mode = "label";
        configure_hybrid_clause_fixture(r, {0,2}, {0,2,4}, {0,2,1,2}, {0,1}, {1,0});
    }
    r.lowfreq_index.label_offset = {0,4,4};
    r.lowfreq_index.total_rows = 4;
    r.lowfreq_index.sample_ids = {1,3,5,7};
    r.lowfreq_index.row_filter_mask = {1,1,1,1};
    if (posting) r.lowfreq_index.label_size = {4,0,0};
    return r;
}

static void check_union_oracle(GraphRuntime r, bool posting, bool grouped, bool deferred = false) {
    validate_and_prepare_queries(r, 2, "greedy_min_posting", {}, deferred);
    require(r.route_to_original_query.size() == 2, "fixture must emit two bounded clause routes");
    const auto selected = r.route_clause_ids;
    const auto old = r;
    require(coalesce_graph_routes(r) == 1, "same producer was not shared");
    require(r.route_to_original_query.size() == 1 && r.route_clause_ids == selected,
            "union lost selected clause ownership");
    require(r.start_local_ids[0] == old.start_local_ids[0], "first selected seed changed");
    require(r.or_route_planner_profile.same_producer_merged_routes == 1,
            "physical merge telemetry missing");
    require(coalesce_graph_routes(r) == 0 && r.or_route_planner_profile.same_producer_merged_routes == 1,
            "finalization is not idempotent");
    materialize_graph_route_payloads(r);
    const auto plan = go_fann::route_predicate::compile(r, posting
        ? go_fann::route_predicate::Producer::LabelPosting
        : go_fann::route_predicate::Producer::Graph);
    require(plan.routes[0].available, "shared producer predicate unavailable");
    if (grouped) {
        require(r.query_filter_bits[0] == 3 && r.query_ranges.modes[0] == 0,
                "group masks were not unioned or flat range was not widened");
    }
    for (uint32_t mask = 0; mask < 8; ++mask) {
        const bool a = mask & 1, b = mask & 2, c = mask & 4;
        const bool expected = grouped ? ((a && c && mask % 2) || (b && c)) : (a && (b || c));
        require(sample_matches_route_predicate(r, 0, mask) == expected,
                "CPU union differs from independent original DNF");
        require(sample_matches_lowfreq_full_labels(r, 0, mask) == expected,
                "CPU final safety helper lost clause-scoped ranges");
        if (!posting) {
            GraphNodeView view; view.runtime = &r;
            const uint64_t producer_mask = grouped ? (mask & 3) : (mask & 1);
            require((compute_host_filter_result(view, mask, producer_mask, 0, true) == 1) == expected,
                    "CPU graph service used a representative conjunct instead of owned OR");
        }
        const bool in_producer = grouped ? (a || b) : a;
        if (in_producer) require(go_fann::route_predicate::evaluate(plan, r, 0, mask) == expected,
                                 "GPU plan oracle differs on producer candidates");
    }
    const auto state = r.route_clause_ids;
    GraphQueryState saved;
    swap_graph_query_state(r, saved);
    swap_graph_query_state(r, saved);
    require(r.route_clause_ids == state && r.query_vectors.rows == 1,
            "online state swap lost shared ownership");
}

static void test_bound_and_domains() {
    auto r = union_fixture();
    // Four branches retain two under the EXISTING fuzzy cap; merging must not
    // re-introduce discarded branches or let freed physical slots expand it.
    configure_hybrid_clause_fixture(r, {0,4}, {0,2,4,6,8},
                                     {0,1,0,2,0,1,0,2}, {0,0,0,0}, {0,0,0,0});
    auto& clauses = r.original_query_clauses;
    clauses.branch_caps = {2}; clauses.branch_cap_is_fuzzy = {1};
    for (size_t i = 1; i < clauses.labels.size(); i += 2) {
        clauses.source_tiers[i] = tier(QueryLabelSourceTier::Direct);
        clauses.relevance_ranks[i] = i;
    }
    validate_and_prepare_queries(r, 2, "greedy_min_posting");
    const auto selected = r.route_clause_ids;
    require(selected.size() == 2, "existing bounded selection did not retain two clauses");
    const auto emitted = r.or_route_planner_profile.emitted_routes;
    coalesce_graph_routes(r);
    require(r.route_clause_ids == selected && r.or_route_planner_profile.emitted_routes == emitted,
            "union changed bounded selection or its telemetry");

    auto distinct = union_fixture();
    validate_and_prepare_queries(distinct, 2, "greedy_min_posting");
    distinct.route_execution_modes[1] = QueryRouteExecution::HostCandidateStream;
    distinct.route_candidate_sources[1] = QueryCandidateSource::LabelPosting;
    require(coalesce_graph_routes(distinct) == 0, "distinct producer execution domains merged");
    auto invalid = union_fixture();
    validate_and_prepare_queries(invalid, 2, "greedy_min_posting");
    invalid.route_to_original_query[1] = 1;
    require_throws_contains([&] { coalesce_graph_routes(invalid); }, "original query domain",
                            "cross-query clause ownership was accepted");

    auto duplicate = union_fixture();
    validate_and_prepare_queries(duplicate, 2, "greedy_min_posting");
    duplicate.route_clause_ids[1] = duplicate.route_clause_ids[0];
    coalesce_graph_routes(duplicate);
    require(duplicate.route_clause_ids.size() == 1, "duplicate owned clauses not deduplicated");
}

static void test_disabled_and_numeric() {
    auto r = union_fixture();
    configure_hybrid_clause_fixture(r, {0,3}, {0,2,4,6},
                                    {0,1,0,2,0,1}, {0,1,2}, {0,0,0});
    validate_and_prepare_queries(r, 2, "greedy_min_posting");
    require(r.query_vectors.rows == 3, "disabled fixture route count");
    r.route_enabled[1] = 0;
    r.high_query_indices = {0,2};
    const auto disabled_clause = r.route_to_clause[1];
    require(coalesce_graph_routes(r) == 1 && r.query_vectors.rows == 2,
            "union must compact only its own duplicates");
    require(r.route_enabled[1] == 0 && r.route_to_clause[1] == disabled_clause &&
            r.high_query_indices == std::vector<uint32_t>{0},
            "preexisting disabled provider metadata was lost");

    auto numeric = union_fixture();
    numeric.supported_query_route_providers |= kQueryRouteProviderRangePosting;
    configure_hybrid_clause_fixture(numeric, {0,1}, {0,0}, {}, {0}, {1});
    validate_and_prepare_queries(numeric, 2, "greedy_min_posting");
    numeric.data_labels = {}; // Numeric-only queries need no label CSR.
    require(sample_matches_route_predicate(numeric, 0, 1) &&
            !sample_matches_route_predicate(numeric, 0, 0),
            "numeric-only DNF incorrectly requires label CSR");
    numeric.route_clause_indptr = {0,0}; numeric.route_clause_ids.clear();
    require(sample_matches_route_predicate(numeric, 0, 1) &&
            !sample_matches_route_predicate(numeric, 0, 0),
            "ordinary numeric residual incorrectly requires label CSR");
}

int main() {
    ScopedRouteCaps caps;
    configure_or_route_planner("legacy");
    check_union_oracle(union_fixture(), false, false);
    check_union_oracle(union_fixture(true), true, false);
    check_union_oracle(union_fixture(false, true), false, true);
    check_union_oracle(union_fixture(), false, false, true);
    check_union_oracle(union_fixture(true), true, false, true);
    check_union_oracle(union_fixture(false, true), false, true, true);
    for (bool posting : {false, true}) {
        auto lazy = union_fixture(posting);
        configure_query_preparation({1, true});
        const auto parsed = evaluate_hybrid_query_manifest_row_for_test(
            hybrid_manifest_row(hybrid_label_atom("categorical", "single", 0), hybrid_fuzzy_atom(), "and"),
            {1, 2}, {1, 1}, {0, 0}, 3, true);
        lazy.original_query_clauses = parsed.clauses;
        check_union_oracle(lazy, posting, false, true);
    }
    test_bound_and_domains();
    test_disabled_and_numeric();
    std::puts("Bounded same-producer route union tests passed");
}
