#include "route_predicate_fixture.h"
#include "execution/async_io.h"

int main()
{
    auto r = predicate_fixture();
    predicate_clauses(r, {{1, 2}}, {0});
    r.query_vectors.rows = 2;
    r.route_to_original_query = {0, 0};
    r.query_group_ids = {0, 0};
    r.query_filter_labels = {1, 1};
    r.query_filter_bits = {1, 1};
    r.query_is_lowfreq = {0, 0};
    r.route_enabled = {1, 1};
    r.route_execution_modes.assign(2, QueryRouteExecution::GraphTraversal);
    r.high_query_indices = {0, 1};
    r.route_clause_indptr = {0, 1, 2};
    r.route_clause_ids = {0, 0};
    r.query_ranges = {2, 2, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    auto plan = go_fann::route_predicate::compile(r);
    GraphNodeView view;
    view.runtime = &r;
    AsyncIoConfig config;
    config.predicate_plan = &plan;
    std::vector<uint8_t> verified{1, 0};
    config.predicate_gpu_verified_routes = &verified;
    predicate_expect(compute_host_filter_result(view, 2, 1, 0, true) == 2,
                     "legacy predicate reference did not reject missing token");
    predicate_expect(compute_host_filter_result(view, 2, 1, 0, true, &config) == 1,
                     "verified GPU route did not return adjacency readiness");
    predicate_expect(compute_host_filter_result(view, 2, 1, 1, true, &config) == 2,
                     "GPU bypass leaked into CPU residual route");
    predicate_expect(compute_host_filter_result(view, 0, 1, 1, true, &config) == 1,
                     "CPU simplified route rejected valid residual");
    verified[0] = 0;
    predicate_expect(compute_host_filter_result(view, 2, 1, 0, true, &config) == 2,
                     "cpu_simplified mode skipped real residual");
    plan.routes[1].available = false;
    predicate_expect(compute_host_filter_result(view, 2, 1, 1, true, &config) == 2,
                     "uncompiled route failed to retain CPU fallback");
    verified[1] = 1;
    bool rejected = false;
    try { host_route_predicate_gpu_verified(config, 1); }
    catch (const std::runtime_error&) { rejected = true; }
    predicate_expect(rejected, "uncompiled route accepted GPU trust");
}
