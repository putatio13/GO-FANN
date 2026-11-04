#include "route_predicate_fixture.h"

namespace rp = go_fann::route_predicate;

int main()
{
    {
        auto r = predicate_fixture();
        auto plan = rp::compile(r);
        predicate_expect(plan.routes[0].entailed, "single identity route must be entailed");
        r.data_labels.indptr.clear(); // Empty residual never needs a sample CSR span.
        predicate_expect(rp::evaluate(plan, r, 0, 2), "entailed route accessed sample CSR");
        predicate_expect(!rp::evaluate(plan, r, 0, 4), "entailed route accepted invalid sample");
    }
    {
        auto r = predicate_fixture();
        predicate_clauses(r, {{1, 2}, {1, 3}}, {0, 0});
        const auto plan = rp::compile(r);
        predicate_expect(plan.routes[0].clauses[0].labels == std::vector<uint32_t>{2} &&
                         plan.routes[0].clauses[1].labels == std::vector<uint32_t>{3},
                         "shared anchor DNF must retain B OR C");
        for (uint32_t sample = 0; sample < 3; ++sample) {
            predicate_expect(rp::evaluate(plan, r, 0, sample) ==
                             sample_matches_route_clause_labels(r, 0, sample),
                             "A AND fuzzy residual differs from original CSR");
        }
        predicate_expect(!rp::evaluate(plan, r, 0, 2), "failed bridge admitted by residual");
    }
    {
        auto r = predicate_fixture();
        predicate_clauses(r, {{1}, {3}}, {0, 0});
        const auto plan = rp::compile(r);
        predicate_expect(plan.routes[0].entailed, "pure fuzzy/OR anchor must imply query");
    }
    {
        auto r = predicate_fixture();
        r.original_query_ranges = {1, 2, {3, 1}, {10, 5}, {30, 0}};
        r.query_ranges = r.original_query_ranges;
        predicate_clauses(r, {{1}, {}}, {0, 1});
        const auto plan = rp::compile(r);
        predicate_expect(plan.routes[0].entailed && rp::evaluate(plan, r, 0, 2),
                         "A OR range must not AND the range onto an entailed label branch");
        predicate_clauses(r, {{1}}, {1});
        const auto mixed = rp::compile(r);
        predicate_expect(mixed.routes[0].clauses[0].labels.empty(), "A AND range retained anchor");
        predicate_expect(!rp::evaluate(mixed, r, 0, 0), "strict numeric lower bound accepted equality");
        predicate_expect(rp::evaluate(mixed, r, 0, 1), "interior mixed-range match rejected");
        predicate_expect(!rp::evaluate(mixed, r, 0, 2), "exclusive upper bound accepted equality");
    }
    {
        auto primary = predicate_fixture();
        primary.original_query_count = 2;
        primary.original_query_ranges = {2, 2, {3, 0, 3, 0}, {100, 0, 10, 0}, {200, 0, 30, 0}};
        auto& t = primary.original_query_clauses;
        t.rows = 2;
        t.query_indptr = {0, 1, 3};
        t.label_indptr = {0, 1, 2, 3};
        t.labels = {1, 2, 3};
        t.top_branch_ids = {0, 0, 1};
        t.uses_original_range = {1, 1, 1};
        primary.range_graph_clause_indptr = {0, 1, 2};
        primary.range_graph_clause_ids = {0, 1}; // Query 1's token 3 branch belongs elsewhere.
        auto secondary = predicate_fixture();
        secondary.original_query_count = 2;
        secondary.route_to_original_query = {1};
        secondary.query_filter_labels = {2};
        secondary.group_layout.groups = {{2}};
        secondary.group_layout.label_to_group = {UINT32_MAX, UINT32_MAX, 0, UINT32_MAX};
        secondary.data_labels.indices.assign(7, 0); // Virtual-bin CSR must never become token CSR.
        const auto plan = rp::compile(secondary, rp::Producer::Graph, &primary);
        predicate_expect(plan.routes[0].clauses.size() == 1 &&
                         plan.routes[0].clauses[0].labels == std::vector<uint32_t>{2},
                         "secondary bin removed equal-valued token or lost clause ownership");
        predicate_expect(rp::evaluate(plan, secondary, 0, 0), "secondary original-query mapping failed");
        predicate_expect(!rp::evaluate(plan, secondary, 0, 1), "secondary accepted unassigned OR branch");
        predicate_expect(!rp::evaluate(plan, primary, 0, 0), "plan accepted wrong graph runtime");
    }
    {
        auto r = predicate_fixture();
        r.label_intervals.has_interval = {0, 1};
        r.query_ranges = {1, 2, {2, 0}, {0, 0}, {20, 0}};
        const auto plan = rp::compile(r);
        predicate_expect(!plan.routes[0].entailed && rp::evaluate(plan, r, 0, 0),
                         "range bin lost residual comparisons");
        predicate_expect(!rp::evaluate(plan, r, 0, 1), "less-than boundary accepted equality");
        r.query_is_lowfreq[0] = 1;
        predicate_expect(!rp::compile(r).routes[0].available, "low-frequency producer inherited graph guarantee");
        r.query_is_lowfreq[0] = 0;
        r.route_execution_modes[0] = QueryRouteExecution::HostCandidateStream;
        predicate_expect(!rp::compile(r).routes[0].available, "posting producer inherited graph guarantee");
        r.route_execution_modes[0] = QueryRouteExecution::GraphTraversal;
        r.group_layout.groups[0].push_back(2);
        const auto grouped_range = rp::compile(r);
        predicate_expect(grouped_range.routes[0].available &&
                         grouped_range.routes[0].clauses[0].labels.empty() &&
                         !rp::evaluate(grouped_range, r, 0, 1),
                         "grouped virtual bin lost numeric predicate or became a token");
    }
    {
        auto r = predicate_fixture();
        predicate_clauses(r, {{1, 2}}, {0});
        r.route_clause_ids[0] = 99;
        bool rejected = false;
        try { rp::compile(r); } catch (const std::runtime_error&) { rejected = true; }
        predicate_expect(rejected, "malformed clause ownership accepted");
    }
    {
        auto r = predicate_fixture();
        r.group_layout.groups = {{1, 2}};
        r.group_layout.label_to_group[2] = 0;
        auto plan = rp::compile(r);
        predicate_expect(plan.routes[0].available && !plan.routes[0].entailed &&
                         plan.routes[0].clauses[0].labels == std::vector<uint32_t>{1},
                         "grouped ordinary route must restore its anchor");
        predicate_expect(rp::evaluate(plan, r, 0, 0) && !rp::evaluate(plan, r, 0, 3),
                         "union graph membership was treated as anchor membership");
        r.query_labels_matrix.indptr = {0, 1};
        r.query_labels_matrix.indices = {2};
        plan = rp::compile(r);
        predicate_expect(plan.routes[0].clauses[0].labels == std::vector<uint32_t>({1, 2}) &&
                         rp::evaluate(plan, r, 0, 0) && !rp::evaluate(plan, r, 0, 1) &&
                         !rp::evaluate(plan, r, 0, 3),
                         "grouped conjunction must check anchor AND residual");
        predicate_clauses(r, {{1, 2}, {3}}, {0, 0});
        plan = rp::compile(r);
        for (uint32_t sample = 0; sample < r.base_vectors.rows; ++sample) {
            predicate_expect(rp::evaluate(plan, r, 0, sample) ==
                             sample_matches_route_clause_labels(r, 0, sample),
                             "grouped DNF differs from full original clauses");
        }
        predicate_expect(rp::evaluate(plan, r, 0, 3),
                         "anchor incorrectly ANDed onto another OR branch");
        r.original_query_ranges = {1, 2, {2, 0}, {0, 0}, {20, 0}};
        predicate_clauses(r, {{1}, {3}}, {1, 0});
        plan = rp::compile(r);
        predicate_expect(rp::evaluate(plan, r, 0, 0) && rp::evaluate(plan, r, 0, 3) &&
                         !rp::evaluate(plan, r, 0, 2),
                         "grouped DNF lost clause-scoped ranges");
        r.group_layout.groups[0] = {2, 3};
        predicate_expect(!rp::compile(r).routes[0].available,
                         "stale group mapping accepted absent anchor");
        r.group_layout.groups[0] = {1, 2};
        r.group_layout.label_to_group[1] = UINT32_MAX;
        predicate_expect(!rp::compile(r).routes[0].available,
                         "inconsistent label-to-group mapping accepted");
    }

}
