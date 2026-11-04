#pragma once

#include "data/dataloader.h"
#include "predicates/range_filter.cuh"

#include <stdexcept>

namespace go_fann::route_predicate {

struct RangeAtom {
    uint32_t dimension = 0;
    uint32_t mode = 0;
    float lower = 0;
    float upper = 0;
};

struct Clause {
    std::vector<uint32_t> labels;
    std::vector<RangeAtom> ranges;
};

struct Route {
    bool available = false;
    bool entailed = false;
    uint32_t original_query = UINT32_MAX;
    std::vector<Clause> clauses;
};

// Borrowed runtime identities outlive all service and rerank workers using a plan.
// A plan proves predicates only for candidates from its selected producer.
// graph_source identifies the selected producer.
struct Plan {
    std::vector<Route> routes;
    const GraphRuntime* label_source = nullptr;
    const GraphRuntime* graph_source = nullptr;
};

inline void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(std::string("route predicate: ") + message);
}

inline std::vector<RangeAtom> compile_ranges(const QueryRangeTable& table,
                                            uint32_t query,
                                            uint32_t dimensions)
{
    require(query < table.rows && table.range_count == dimensions,
            "query range domain mismatch");
    const size_t begin = static_cast<size_t>(query) * dimensions;
    const size_t end = begin + dimensions;
    require(end <= table.modes.size() && end <= table.lower_bounds.size() &&
                end <= table.upper_bounds.size(), "invalid range buffers");
    std::vector<RangeAtom> atoms;
    for (uint32_t dim = 0; dim < dimensions; ++dim) {
        const auto mode = table.modes[begin + dim];
        require(mode <= 3, "unknown range mode");
        if (mode != 0) {
            atoms.push_back({dim, mode, table.lower_bounds[begin + dim],
                             table.upper_bounds[begin + dim]});
        }
    }
    return atoms;
}

enum class Producer { Graph, LabelPosting, RangePosting };

namespace detail {

// Ownership and lowering are shared by every producer. The selected producer
// only establishes which routes/clauses it owns and which anchor it proves.
inline std::vector<uint32_t> owned_clauses(const std::vector<uint64_t>& offsets,
                                          const std::vector<uint32_t>& ids,
                                          size_t row_count, uint32_t row)
{
    require(offsets.size() == row_count + 1, "invalid clause ownership table length");
    const auto begin = offsets[row], end = offsets[row + 1];
    require(begin <= end && end <= ids.size(), "invalid clause ownership offsets");
    return {ids.begin() + begin, ids.begin() + end};
}

inline void append_clauses(Route& route, const GraphRuntime& source,
                           const std::vector<uint32_t>& ids,
                           uint32_t anchor, bool anchor_guaranteed, bool posting)
{
    const auto& table = source.original_query_clauses;
    require((!posting || table.query_indptr.size() ==
                static_cast<size_t>(source.original_query_count) + 1) &&
                table.has_clauses(route.original_query) &&
                table.descriptor_shape_valid(),
            "invalid original clause table");
    const auto first = table.query_indptr[route.original_query];
    const auto last = table.query_indptr[route.original_query + 1];
    require(first <= last && last <= table.clause_count(), "invalid query clause offsets");
    for (const uint32_t id : ids) {
        require(id >= first && id < last, "clause belongs to another original query");
        const auto labels = table.clause_labels(id, route.original_query);
        require(labels.valid, "invalid clause labels");
        Clause clause;
        for (auto it = labels.begin(); it != labels.end(); ++it) {
            const uint32_t label = *it;
            if (!anchor_guaranteed || label != anchor) clause.labels.push_back(label);
        }
        if (table.clause_uses_range(id, route.original_query)) {
            clause.ranges = compile_ranges(source.original_query_ranges,
                                            route.original_query, source.base_range_count);
        }
        route.clauses.push_back(std::move(clause));
    }
}

inline void append_ordinary_clause(Route& route, const SparseLabelRowSpan& labels,
                                   uint32_t anchor, bool restore_anchor,
                                   const QueryRangeTable& ranges, uint32_t range_row,
                                   uint32_t dimensions)
{
    Clause clause;
    if (restore_anchor) clause.labels.push_back(anchor);
    for (auto it = labels.begin; it != labels.end; ++it) {
        if (*it != anchor) clause.labels.push_back(*it);
    }
    clause.ranges = compile_ranges(ranges, range_row, dimensions);
    route.clauses.push_back(std::move(clause));
}

inline void finish(Route& route, const GraphRuntime& source, bool posting)
{
    for (const auto& clause : route.clauses) {
        if (posting) {
            for (uint32_t label : clause.labels) {
                require(label < source.data_labels.cols, "posting token outside original-label domain");
            }
        }
        if (clause.labels.empty() && clause.ranges.empty()) route.entailed = true;
    }
    route.available = true;
    if (route.entailed) route.clauses.assign(1, Clause{});
}

inline bool resident(QueryRouteExecution execution)
{
    return execution == QueryRouteExecution::GpuResidentScan ||
           execution == QueryRouteExecution::GpuResidentScanPostFilter;
}

inline bool posting_proves_anchor(const GraphRuntime& runtime,
                                   uint32_t route, uint32_t anchor)
{
    // The posting builder uses the same GroupLayout as graph construction,
    // but membership is proved by this producer's own list, not graph rows.
    // A multi-label list is a union: its optional row masks are not a proof
    // shared by every posting scan, so retain the anchor in that case.
    const auto& layout = runtime.group_layout;
    const auto& index = runtime.lowfreq_index;
    if (anchor >= runtime.data_labels.cols ||
        runtime.query_group_ids.size() != runtime.query_vectors.rows ||
        anchor >= layout.label_to_group.size() ||
        index.group_count != layout.groups.size() ||
        index.label_offset.size() != index.group_count ||
        index.label_size.size() != index.group_count ||
        index.sample_ids.size() != index.total_rows ||
        index.row_filter_mask.size() != index.total_rows) return false;
    const uint32_t group = runtime.query_group_ids[route];
    if (group >= index.group_count || layout.label_to_group[anchor] != group ||
        layout.groups[group].size() != 1 || layout.groups[group][0] != anchor) return false;
    const uint64_t begin = index.label_offset[group];
    const uint64_t count = index.label_size[group];
    return count != 0 && begin + count <= index.total_rows;
}

} // namespace detail

// Range representatives are output slots, not candidate-source proofs. Repeated
// range slices may share a representative; its original query must still match.
// Label postings may prove singleton anchors; route-owned OR scope is unchanged.
inline Plan compile(const GraphRuntime& runtime,
                    Producer producer = Producer::Graph,
                    const GraphRuntime* primary = nullptr,
                    const std::vector<uint32_t>& range_representatives = {},
                    const std::vector<uint32_t>& range_original_queries = {})
{
    Plan plan;
    plan.graph_source = &runtime;
    plan.label_source = primary != nullptr ? primary : &runtime;
    const GraphRuntime& source = *plan.label_source;
    const bool posting = producer != Producer::Graph;
    const size_t route_count = runtime.query_vectors.rows;
    plan.routes.resize(route_count);
    if (!posting) {
        require(runtime.route_to_original_query.size() == route_count &&
                    runtime.query_group_ids.size() == route_count &&
                    runtime.query_filter_labels.size() == route_count,
                "route metadata is not aligned");
        if (primary != nullptr) {
            require(source.original_query_count == runtime.original_query_count &&
                        source.base_vectors.rows == runtime.base_vectors.rows &&
                        source.base_range_count == runtime.base_range_count,
                    "secondary graph sample/query domain differs from primary");
        }
    } else {
        require(runtime.route_to_original_query.size() == route_count &&
                    runtime.route_providers.size() == route_count &&
                    runtime.route_execution_modes.size() == route_count &&
                    runtime.route_candidate_sources.size() == route_count &&
                    (runtime.route_enabled.empty() || runtime.route_enabled.size() == route_count),
                "posting route metadata is not aligned");
        require(source.original_query_count == runtime.original_query_count &&
                    source.base_vectors.rows == runtime.base_vectors.rows &&
                    source.data_labels.rows == runtime.base_vectors.rows &&
                    source.base_range_count == runtime.base_range_count,
                "posting original-label sample/query domain mismatch");
        if (producer == Producer::LabelPosting) {
            require(runtime.query_is_lowfreq.size() == route_count &&
                        runtime.query_filter_labels.size() == route_count,
                    "posting label route metadata is not aligned");
        }
    }
    if (producer == Producer::RangePosting) {
        require(range_representatives.size() == range_original_queries.size(),
                "posting range maps differ in size");
    }
    std::vector<uint8_t> seen(producer == Producer::RangePosting ? route_count : 0, 0);
    const size_t selected_count = producer == Producer::Graph ? runtime.high_query_indices.size() :
        (producer == Producer::RangePosting ? range_representatives.size() : route_count);
    for (size_t pos = 0; pos < selected_count; ++pos) {
        const uint32_t id = producer == Producer::Graph ? runtime.high_query_indices[pos] :
            (producer == Producer::RangePosting ? range_representatives[pos] : pos);
        if (producer == Producer::RangePosting) {
            const auto original = range_original_queries[pos];
            require(id < route_count && original < source.original_query_count &&
                        runtime.route_to_original_query[id] == original,
                    "posting representative/query identity mismatch");
            if (seen[id]) continue;
            seen[id] = 1;
        } else {
            require(id < route_count, "graph route index out of bounds");
        }
        if (!runtime.route_enabled.empty() && runtime.route_enabled.at(id) == 0) continue;
        uint32_t anchor = UINT32_MAX;
        bool virtual_anchor = false;
        bool anchor_guaranteed = false;
        if (producer == Producer::Graph) {
            if ((!runtime.route_execution_modes.empty() &&
                 runtime.route_execution_modes.at(id) != QueryRouteExecution::GraphTraversal) ||
                (!runtime.query_is_lowfreq.empty() && runtime.query_is_lowfreq.at(id) != 0)) continue;
            const uint32_t group = runtime.query_group_ids[id];
            anchor = runtime.query_filter_labels[id];
            if (group >= runtime.group_layout.groups.size() ||
                anchor >= runtime.group_layout.label_to_group.size() ||
                runtime.group_layout.label_to_group[anchor] != group) continue;
            const auto& group_labels = runtime.group_layout.groups[group];
            if (std::find(group_labels.begin(), group_labels.end(), anchor) == group_labels.end()) continue;
            virtual_anchor = primary != nullptr ||
                (anchor < runtime.label_intervals.has_interval.size() &&
                 runtime.label_intervals.has_interval[anchor] != 0);
            anchor_guaranteed = !virtual_anchor && group_labels.size() == 1;
        } else if (producer == Producer::LabelPosting) {
            if (runtime.query_is_lowfreq[id] == 0 ||
                runtime.route_providers[id] != QueryRouteProvider::Label ||
                runtime.route_candidate_sources[id] != QueryCandidateSource::LabelPosting ||
                !detail::resident(runtime.route_execution_modes[id])) continue;
            anchor = runtime.query_filter_labels[id];
            virtual_anchor = primary != nullptr ||
                (anchor < runtime.label_intervals.has_interval.size() &&
                 runtime.label_intervals.has_interval[anchor] != 0);
            anchor_guaranteed = !virtual_anchor &&
                detail::posting_proves_anchor(runtime, id, anchor);
        }

        Route& route = plan.routes[id];
        route.original_query = runtime.route_to_original_query[id];
        require(route.original_query < source.original_query_count, "original query index out of bounds");
        const bool range_replacement = producer == Producer::RangePosting &&
            runtime.route_providers[id] != QueryRouteProvider::RangePosting;
        if (producer == Producer::RangePosting && !range_replacement &&
            (!detail::resident(runtime.route_execution_modes[id]) ||
             runtime.route_candidate_sources[id] != QueryCandidateSource::RangePosting)) continue;

        std::vector<uint32_t> clause_ids;
        if (producer == Producer::Graph && primary != nullptr) {
            // A secondary bin is never an original token, even if IDs collide.
            clause_ids = detail::owned_clauses(source.range_graph_clause_indptr,
                source.range_graph_clause_ids, source.original_query_count, route.original_query);
            if (clause_ids.empty()) continue;
        } else if (!range_replacement && !runtime.route_clause_indptr.empty()) {
            clause_ids = detail::owned_clauses(runtime.route_clause_indptr,
                runtime.route_clause_ids, route_count, id);
        }
        if (!clause_ids.empty()) {
            detail::append_clauses(route, source, clause_ids, anchor, anchor_guaranteed, posting);
        } else {
            // Missing posting DNF ownership cannot be reconstructed from a
            // representative route. Keep graph's existing ordinary fallback.
            if (posting && source.original_query_clauses.has_clauses(route.original_query)) continue;
            if (producer == Producer::RangePosting) {
                const auto labels = sparse_label_row_span(source.original_query_labels_matrix,
                                                           route.original_query);
                if (!labels.valid || labels.begin != labels.end) continue;
                const auto& ranges = range_replacement ? source.original_query_ranges : runtime.query_ranges;
                const auto row = range_replacement ? route.original_query : id;
                detail::append_ordinary_clause(route, labels, anchor, false, ranges, row,
                                                runtime.base_range_count);
            } else {
                if (producer == Producer::LabelPosting && virtual_anchor) continue;
                const auto labels = sparse_label_row_span(runtime.query_labels_matrix, id);
                require(labels.valid, "invalid ordinary route residual CSR");
                if (virtual_anchor && labels.begin != labels.end) continue;
                detail::append_ordinary_clause(route, labels, anchor,
                    !virtual_anchor && !anchor_guaranteed, runtime.query_ranges, id,
                    runtime.base_range_count);
            }
        }
        detail::finish(route, source, posting);
    }
    return plan;
}

inline bool evaluate(const Plan& plan, const GraphRuntime& runtime,
                     uint32_t route_idx, uint32_t sample)
{
    if (plan.graph_source != &runtime || plan.label_source == nullptr ||
        route_idx >= plan.routes.size() || !plan.routes[route_idx].available ||
        sample >= runtime.base_vectors.rows) return false;
    const Route& route = plan.routes[route_idx];
    if (route.entailed) return true;
    SparseLabelRowSpan labels;
    bool labels_loaded = false;
    for (const Clause& clause : route.clauses) {
        bool pass = true;
        if (!clause.labels.empty()) {
            if (!labels_loaded) {
                labels = sparse_label_row_span(plan.label_source->data_labels, sample);
                labels_loaded = true;
            }
            pass = labels.valid;
            for (uint32_t label : clause.labels) {
                if (!pass || labels.begin == labels.end ||
                    !std::binary_search(labels.begin, labels.end, label)) {
                    pass = false;
                    break;
                }
            }
        }
        if (!pass) continue;
        for (const RangeAtom& atom : clause.ranges) {
            const size_t flat = static_cast<size_t>(sample) * runtime.base_range_count + atom.dimension;
            if (atom.dimension >= runtime.base_range_count || flat >= runtime.base_range_values.size() ||
                !range_filter::range_filter_matches_single(static_cast<uint8_t>(atom.mode),
                    atom.lower, atom.upper, runtime.base_range_values[flat])) {
                pass = false;
                break;
            }
        }
        if (pass) return true;
    }
    return false;
}

} // namespace go_fann::route_predicate
