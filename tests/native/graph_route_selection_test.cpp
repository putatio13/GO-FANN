#include "data/dataloader.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

GraphRuntime make_runtime()
{
    GraphRuntime runtime;
    runtime.vector_scalar_type = DenseVectorScalarType::Uint8;
    runtime.original_query_count = 2;
    runtime.original_query_label_ops.assign(2, QueryLabelOp::And);
    runtime.original_query_labels_matrix.rows = 2;
    runtime.original_query_labels_matrix.cols = 1;
    runtime.original_query_labels_matrix.indptr = {0, 0, 0};

    runtime.metadata.label_count = 1;
    runtime.metadata.label_size = {3};
    runtime.metadata.label_offset = {0};
    runtime.group_layout.groups = {{0}};
    runtime.group_layout.label_to_group = {0};

    runtime.query_vectors.rows = 3;
    runtime.query_vectors.cols = 2;
    runtime.query_vectors.data = {1, 2, 3, 4, 5, 6};
    runtime.query_labels_matrix.rows = 3;
    runtime.query_labels_matrix.cols = 1;
    runtime.query_labels_matrix.indptr = {0, 0, 0, 0};
    runtime.query_labels_matrix.row_labels.resize(3);
    runtime.query_ranges.rows = 3;
    runtime.query_ranges.range_count = 1;
    runtime.query_ranges.modes = {1, 2, 3};
    runtime.query_ranges.lower_bounds = {1.0f, 2.0f, 3.0f};
    runtime.query_ranges.upper_bounds = {4.0f, 5.0f, 6.0f};

    runtime.query_group_ids = {0, 0, 0};
    runtime.query_filter_labels = {0, 0, 0};
    runtime.query_filter_bits = {0, 0, 0};
    runtime.start_local_ids = {0, 1, 2};
    runtime.query_is_lowfreq = {0, 0, 0};
    runtime.route_to_original_query = {0, 1, 1};
    runtime.route_to_clause = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    runtime.route_clause_indptr = {0, 0, 0, 0};
    runtime.route_top_branch_ids = {0, 0, 0};
    runtime.route_enabled = {1, 1, 1};
    runtime.route_providers.assign(3, QueryRouteProvider::Label);
    runtime.route_execution_modes.assign(3, QueryRouteExecution::GraphTraversal);
    runtime.route_candidate_sources.assign(3, QueryCandidateSource::None);
    runtime.high_query_indices = {0, 1, 2};
    return runtime;
}

void test_route_selection()
{
    auto runtime = make_runtime();
    const size_t removed = retain_graph_routes_for_original_queries(
        runtime, std::vector<uint8_t>{0, 1});
    require(removed == 1, "wrong removed route count");
    require(runtime.original_query_count == 2, "original query count changed");
    require(runtime.route_to_original_query == std::vector<uint32_t>({1, 1}),
            "route map was not retained");
    require(runtime.query_vectors.data == std::vector<uint8_t>({3, 4, 5, 6}),
            "routed query vectors were not compacted");
    require(runtime.query_ranges.modes == std::vector<uint8_t>({2, 3}),
            "range rows were not compacted");
    require(runtime.query_labels_matrix.indptr == std::vector<uint64_t>({0, 0, 0}),
            "label CSR was not compacted");
    require(runtime.high_query_indices == std::vector<uint32_t>({0, 1}),
            "high route indexes were not rebuilt");
}

void test_hybrid_range_branch_residual_labels()
{
    GraphRuntime runtime;
    runtime.data_labels.rows = 4;
    runtime.data_labels.cols = 3;
    runtime.data_labels.indptr = {0, 1, 2, 4, 4};
    runtime.data_labels.indices = {1, 2, 1, 2};

    HybridQueryClauseTable& clauses = runtime.original_query_clauses;
    runtime.original_query_count = 2;
    clauses.rows = 2;
    clauses.query_indptr = {0, 2, 4};
    clauses.label_indptr = {0, 1, 2, 2, 3};
    clauses.labels = {1, 2, 1};
    clauses.source_tiers.assign(clauses.labels.size(), 0);
    clauses.relevance_ranks.assign(clauses.labels.size(), 0);
    clauses.top_branch_ids = {0, 0, 0, 1};
    clauses.uses_original_range = {1, 1, 1, 0};

    require(sample_matches_hybrid_range_branch_labels(runtime, 0, 0),
            "first fuzzy DNF label should match");
    require(sample_matches_hybrid_range_branch_labels(runtime, 0, 1),
            "second fuzzy DNF label should match");
    require(sample_matches_hybrid_range_branch_labels(runtime, 0, 2),
            "multi-label sample should match either fuzzy DNF clause");
    require(!sample_matches_hybrid_range_branch_labels(runtime, 0, 3),
            "empty sample must not match a labeled range branch");
    require(sample_matches_hybrid_range_branch_labels(runtime, 1, 3),
            "standalone range OR branch must have an empty label residual");

    runtime.range_graph_clause_indptr = {0, 1, 1};
    runtime.range_graph_clause_ids = {1};
    require(!sample_matches_range_graph_clause_labels(runtime, 0, 0) &&
                sample_matches_range_graph_clause_labels(runtime, 0, 1),
            "range graph residual check included an unassigned range clause");
}

void test_nonbinary_range_cover_coarsening()
{
    std::vector<uint8_t> children(32, 0);
    std::fill(children.begin() + 8, children.begin() + 16, 1);
    children[16] = 1;
    children[17] = 1;
    std::vector<uint8_t> parents(4, 0);
    const std::vector<uint32_t> parent_labels = {100, 101, 102, 103};

    coarsen_range_cover_bins(children, parents, parent_labels);

    require(parents == std::vector<uint8_t>({0, 1, 0, 0}),
            "complete eight-child range cover was not coarsened");
    require(std::all_of(children.begin() + 8, children.begin() + 16,
                        [](uint8_t active) { return active == 0; }),
            "coarsened range children were not cleared");
    require(children[16] == 1 && children[17] == 1,
            "partial sibling range cover was incorrectly coarsened");
}

}  // namespace

int main()
{
    try {
        test_route_selection();
        test_hybrid_range_branch_residual_labels();
        test_nonbinary_range_cover_coarsening();
        std::puts("graph runtime route selection tests passed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "graph runtime route selection test failed: %s\n", error.what());
        return 1;
    }
}
