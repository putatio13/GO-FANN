#pragma once

#include "planning/route_predicate_plan.h"
#include <initializer_list>

inline GraphRuntime predicate_fixture()
{
    GraphRuntime r;
    r.base_vectors.rows = 4;
    r.base_range_count = 2;
    r.base_range_values = {10, 5, 20, 15, 30, 25, 10, 15};
    r.data_labels.rows = 4;
    r.data_labels.cols = 4;
    r.data_labels.indptr = {0, 2, 4, 5, 7};
    r.data_labels.indices = {1, 2, 1, 3, 1, 2, 3};
    r.original_query_count = 1;
    r.query_vectors.rows = 1;
    r.route_to_original_query = {0};
    r.query_group_ids = {0};
    r.query_filter_labels = {1};
    r.query_filter_bits = {1};
    r.query_is_lowfreq = {0};
    r.route_enabled = {1};
    r.route_execution_modes = {QueryRouteExecution::GraphTraversal};
    r.high_query_indices = {0};
    r.group_layout.groups = {{1}};
    r.group_layout.label_to_group = {UINT32_MAX, 0, UINT32_MAX, UINT32_MAX};
    r.query_labels_matrix.rows = 1;
    r.query_labels_matrix.indptr = {0, 0};
    r.original_query_ranges = {1, 2, {0, 0}, {0, 0}, {0, 0}};
    r.query_ranges = r.original_query_ranges;
    return r;
}

inline void predicate_clauses(GraphRuntime& r,
                              std::initializer_list<std::vector<uint32_t>> labels,
                              const std::vector<uint8_t>& range_flags)
{
    auto& t = r.original_query_clauses;
    t = {};
    t.rows = 1;
    t.query_indptr = {0, labels.size()};
    t.label_indptr = {0};
    t.uses_original_range = range_flags;
    for (const auto& row : labels) {
        t.labels.insert(t.labels.end(), row.begin(), row.end());
        t.label_indptr.push_back(t.labels.size());
        t.top_branch_ids.push_back(0);
    }
    r.route_clause_indptr = {0, labels.size()};
    r.route_clause_ids.clear();
    for (uint32_t id = 0; id < labels.size(); ++id) r.route_clause_ids.push_back(id);
}

inline void predicate_expect(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}
