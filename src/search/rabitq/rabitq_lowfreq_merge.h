#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace go_fann::backend_rabitq::lowfreq_merge {

using CandidateId = uint64_t;

enum class FinalMergeStrategy {
    SingleRouteCopy,
    SortedRouteHeap,
    HashDedupSort,
};

inline const char* final_merge_strategy_name(FinalMergeStrategy strategy)
{
    switch (strategy) {
        case FinalMergeStrategy::SingleRouteCopy:
            return "single_route_copy";
        case FinalMergeStrategy::SortedRouteHeap:
            return "sorted_route_heap";
        case FinalMergeStrategy::HashDedupSort:
            return "hash_dedup_sort";
    }
    return "unknown";
}

struct RouteGroup {
    uint32_t original_query = UINT32_MAX;
    std::vector<size_t> route_positions;
};

struct Candidate {
    uint32_t sample_id = UINT32_MAX;
    float approximate_distance = std::numeric_limits<float>::max();
    uint32_t source_route_id = UINT32_MAX;
    uint32_t source_rank = UINT32_MAX;
};

struct Diagnostics {
    uint64_t original_query_groups = 0;
    uint64_t route_candidate_slots = 0;
    uint64_t route_valid_candidates = 0;
    uint64_t unique_candidates = 0;
    uint64_t retained_candidates = 0;
    uint64_t exact_vector_reads = 0;
    uint64_t representative_result_rows = 0;
    uint64_t compact_empty_result_rows = 0;
    uint64_t route_step_rows = 0;
    uint64_t sorted_merge_groups = 0;
    uint64_t fallback_merge_groups = 0;
    uint64_t merge_visited_slots = 0;
    // Sorted merging can stop at the cap. Valid/unique counts then describe
    // the consumed prefix, not the unvisited union. Slot capacity stays exact.
    uint64_t incomplete_candidate_count_groups = 0;

    Diagnostics& operator+=(const Diagnostics& other)
    {
        original_query_groups += other.original_query_groups;
        route_candidate_slots += other.route_candidate_slots;
        route_valid_candidates += other.route_valid_candidates;
        unique_candidates += other.unique_candidates;
        retained_candidates += other.retained_candidates;
        exact_vector_reads += other.exact_vector_reads;
        representative_result_rows += other.representative_result_rows;
        compact_empty_result_rows += other.compact_empty_result_rows;
        route_step_rows += other.route_step_rows;
        sorted_merge_groups += other.sorted_merge_groups;
        fallback_merge_groups += other.fallback_merge_groups;
        merge_visited_slots += other.merge_visited_slots;
        incomplete_candidate_count_groups += other.incomplete_candidate_count_groups;
        return *this;
    }
};

template <typename T>
bool ensure_reusable_host_staging_slots(std::vector<T>& buffer, size_t required_slots)
{
    if (buffer.size() >= required_slots) {
        return true;
    }
    buffer.resize(required_slots);
    return false;
}

inline std::vector<RouteGroup> build_route_groups(
    const std::vector<uint32_t>& route_ids,
    const std::vector<uint32_t>& route_to_original_query,
    uint32_t original_query_count)
{
    std::vector<RouteGroup> groups;
    groups.reserve(std::min<size_t>(route_ids.size(), original_query_count));
    std::unordered_map<uint32_t, size_t> group_by_original;
    group_by_original.reserve(groups.capacity());
    for (size_t route_pos = 0; route_pos < route_ids.size(); ++route_pos) {
        const uint32_t route_id = route_ids[route_pos];
        if (route_id >= route_to_original_query.size()) {
            throw std::runtime_error("low-frequency merge route id is outside route_to_original_query");
        }
        const uint32_t original_query = route_to_original_query[route_id];
        if (original_query >= original_query_count) {
            throw std::runtime_error("low-frequency merge original query id is out of range");
        }
        auto [it, inserted] = group_by_original.emplace(original_query, groups.size());
        if (inserted) {
            RouteGroup group;
            group.original_query = original_query;
            groups.push_back(std::move(group));
        }
        groups[it->second].route_positions.push_back(route_pos);
    }
    return groups;
}

inline float normalized_approximate_distance(float distance)
{
    return std::isfinite(distance) ? distance : std::numeric_limits<float>::max();
}

inline bool duplicate_provenance_is_better(float distance,
                                           uint32_t rank,
                                           uint32_t route_id,
                                           const Candidate& current)
{
    if (distance != current.approximate_distance) {
        return distance < current.approximate_distance;
    }
    if (rank != current.source_rank) {
        return rank < current.source_rank;
    }
    return route_id < current.source_route_id;
}

enum class CandidateStage { Collect, Rank };
struct IgnoreCandidateStage {
    void operator()(CandidateStage) const {}
};
struct DirectCandidateHash {
    template <typename Operation>
    decltype(auto) operator()(Operation&& operation) const { return operation(); }
};

template <typename RouteFilter, typename StageObserver = IgnoreCandidateStage,
          typename HashObserver = DirectCandidateHash>
std::vector<Candidate> select_group_candidates(
    const RouteGroup& group,
    const std::vector<uint32_t>& route_ids,
    const std::vector<CandidateId>& route_candidate_ids,
    const std::vector<float>& route_candidate_distances,
    uint32_t route_shortlist_width,
    uint64_t base_row_count,
    uint32_t retained_cap,
    RouteFilter&& route_filter,
    Diagnostics& diagnostics,
    StageObserver observe_stage = {},
    HashObserver observe_hash = {})
{
    if (route_shortlist_width != 0 &&
        route_ids.size() > std::numeric_limits<size_t>::max() / route_shortlist_width) {
        throw std::runtime_error("low-frequency merge route shortlist size overflows size_t");
    }
    const size_t required_slots = route_ids.size() * static_cast<size_t>(route_shortlist_width);
    if (route_candidate_ids.size() < required_slots ||
        route_candidate_distances.size() < required_slots) {
        throw std::runtime_error("low-frequency merge route shortlist buffers are too small");
    }

    if (route_shortlist_width != 0 &&
        group.route_positions.size() > std::numeric_limits<size_t>::max() / route_shortlist_width) {
        throw std::runtime_error("low-frequency merge group shortlist size overflows size_t");
    }
    const size_t group_slot_count =
        group.route_positions.size() * static_cast<size_t>(route_shortlist_width);
    observe_stage(CandidateStage::Collect);
    std::vector<Candidate> unique_candidates;
    unique_candidates.reserve(group_slot_count);
    std::unordered_map<uint32_t, size_t> candidate_by_sample;
    candidate_by_sample.reserve(group_slot_count);

    for (size_t route_pos : group.route_positions) {
        if (route_pos >= route_ids.size()) {
            throw std::runtime_error("low-frequency merge group route position is out of range");
        }
        const uint32_t route_id = route_ids[route_pos];
        for (uint32_t rank = 0; rank < route_shortlist_width; ++rank) {
            ++diagnostics.route_candidate_slots;
            const size_t offset = route_pos * static_cast<size_t>(route_shortlist_width) + rank;
            const CandidateId candidate_id = route_candidate_ids[offset];
            if (candidate_id == UINT32_MAX || candidate_id >= base_row_count) {
                continue;
            }
            const uint32_t sample_id = static_cast<uint32_t>(candidate_id);
            if (!route_filter(route_id, sample_id)) {
                continue;
            }
            ++diagnostics.route_valid_candidates;
            const float distance = normalized_approximate_distance(route_candidate_distances[offset]);
            auto [it, inserted] = observe_hash([&] {
                return candidate_by_sample.emplace(sample_id, unique_candidates.size());
            });
            if (inserted) {
                unique_candidates.push_back({sample_id, distance, route_id, rank});
            } else {
                Candidate& current = unique_candidates[it->second];
                if (duplicate_provenance_is_better(distance, rank, route_id, current)) {
                    current.approximate_distance = distance;
                    current.source_route_id = route_id;
                    current.source_rank = rank;
                }
            }
        }
    }

    diagnostics.unique_candidates += unique_candidates.size();
    observe_stage(CandidateStage::Rank);
    std::sort(unique_candidates.begin(),
              unique_candidates.end(),
              [](const Candidate& lhs, const Candidate& rhs) {
                  if (lhs.approximate_distance != rhs.approximate_distance) {
                      return lhs.approximate_distance < rhs.approximate_distance;
                  }
                  return lhs.sample_id < rhs.sample_id;
              });
    if (unique_candidates.size() > retained_cap) {
        unique_candidates.resize(retained_cap);
    }
    diagnostics.retained_candidates += unique_candidates.size();
    return unique_candidates;
}

struct CandidateMergeCursor {
    Candidate candidate;
    size_t route_position;
};

inline bool candidate_merge_key_less(const Candidate& a, const Candidate& b)
{
    if (a.approximate_distance != b.approximate_distance)
        return a.approximate_distance < b.approximate_distance;
    if (a.sample_id != b.sample_id) return a.sample_id < b.sample_id;
    if (a.source_rank != b.source_rank) return a.source_rank < b.source_rank;
    return a.source_route_id < b.source_route_id;
}

// Worker-private buffers are reused across original queries. The heap holds
// one head per route; the flat set holds at most retained_cap IDs, never RL.
struct CandidateMergeScratch {
    std::vector<CandidateMergeCursor> heads;
    std::vector<uint32_t> seen;

    void reset(size_t routes, size_t cap)
    {
        heads.clear();
        heads.reserve(routes);
        size_t slots = 1;
        while (slots / 2 < cap) {
            if (slots > std::numeric_limits<size_t>::max() / 2)
                throw std::runtime_error("candidate merge set size overflow");
            slots *= 2;
        }
        seen.assign(slots, UINT32_MAX);
    }

    bool insert(uint32_t id)
    {
        size_t slot = (uint64_t(id) * 11400714819323198485ull) & (seen.size() - 1);
        while (seen[slot] != UINT32_MAX) {
            if (seen[slot] == id) return false;
            slot = (slot + 1) & (seen.size() - 1);
        }
        seen[slot] = id;
        return true;
    }
};

// Same candidate/provenance contract as select_group_candidates. Verify row
// order before consuming any predicate callbacks; unsorted/legacy producers
// retain the established selector. Invalid IDs and nonfinite scores follow
// that selector's normalization. No route-count bound is imposed.
template <typename RouteFilter, typename StageObserver = IgnoreCandidateStage,
          typename HashObserver = DirectCandidateHash>
void merge_sorted_group_candidates_into(
    const RouteGroup& group,
    const std::vector<uint32_t>& route_ids,
    const std::vector<CandidateId>& ids,
    const std::vector<float>& distances,
    uint32_t width,
    uint64_t base_rows,
    uint32_t cap,
    RouteFilter&& filter,
    Diagnostics& diagnostics,
    std::vector<Candidate>& retained,
    StageObserver observe_stage = {},
    HashObserver observe_hash = {})
{
    if (width && route_ids.size() > std::numeric_limits<size_t>::max() / width)
        throw std::runtime_error("candidate merge buffer size overflow");
    const size_t required = route_ids.size() * size_t(width);
    if (ids.size() < required || distances.size() < required)
        throw std::runtime_error("candidate merge buffers are too small");
    if (width && group.route_positions.size() > std::numeric_limits<size_t>::max() / width)
        throw std::runtime_error("candidate merge group size overflow");
    observe_stage(CandidateStage::Collect);
    bool sorted = true;
    for (size_t pos : group.route_positions) {
        if (pos >= route_ids.size()) throw std::runtime_error("candidate merge route out of range");
        bool have_previous = false;
        Candidate previous;
        for (uint32_t rank = 0; rank < width; ++rank) {
            const size_t offset = pos * width + rank;
            if (ids[offset] == UINT32_MAX || ids[offset] >= base_rows) continue;
            const Candidate candidate{uint32_t(ids[offset]), normalized_approximate_distance(distances[offset]),
                                      route_ids[pos], rank};
            if (have_previous && candidate_merge_key_less(candidate, previous)) sorted = false;
            previous = candidate;
            have_previous = true;
        }
    }
    if (!sorted) {
        ++diagnostics.fallback_merge_groups;
        diagnostics.merge_visited_slots += group.route_positions.size() * size_t(width);
        retained = select_group_candidates(group, route_ids, ids, distances, width, base_rows,
                                           cap, std::forward<RouteFilter>(filter), diagnostics,
                                           observe_stage, observe_hash);
        return;
    }
    ++diagnostics.sorted_merge_groups;
    diagnostics.route_candidate_slots += group.route_positions.size() * size_t(width);
    // A thread owns this scratch until the synchronous selector returns; no
    // pointer into it escapes to workers or the result vectors.
    thread_local CandidateMergeScratch scratch;
    const size_t limit = std::min<size_t>(cap, group.route_positions.size() * size_t(width));
    scratch.reset(group.route_positions.size(), limit);
    retained.clear();
    retained.reserve(limit);
    const auto advance = [&](size_t pos, uint32_t rank) {
        for (; rank < width; ++rank) {
            ++diagnostics.merge_visited_slots;
            const size_t offset = pos * width + rank;
            if (ids[offset] == UINT32_MAX || ids[offset] >= base_rows) continue;
            const uint32_t sample = uint32_t(ids[offset]);
            if (!filter(route_ids[pos], sample)) continue;
            ++diagnostics.route_valid_candidates;
            scratch.heads.push_back({{sample, normalized_approximate_distance(distances[offset]),
                                     route_ids[pos], rank}, pos});
            return;
        }
    };
    // Do not do predicate work when the requested output is empty.
    if (limit == 0) {
        if (width && !group.route_positions.empty()) ++diagnostics.incomplete_candidate_count_groups;
        return;
    }
    if (group.route_positions.size() == 1) {
        const size_t pos = group.route_positions.front();
        for (uint32_t rank = 0; rank < width; ++rank) {
            ++diagnostics.merge_visited_slots;
            const size_t offset = pos * width + rank;
            if (ids[offset] == UINT32_MAX || ids[offset] >= base_rows) continue;
            const uint32_t sample = uint32_t(ids[offset]);
            if (!filter(route_ids[pos], sample)) continue;
            ++diagnostics.route_valid_candidates;
            if (observe_hash([&] { return scratch.insert(sample); })) {
                retained.push_back({sample, normalized_approximate_distance(distances[offset]),
                                    route_ids[pos], rank});
                ++diagnostics.unique_candidates;
            }
            if (retained.size() == limit) {
                ++diagnostics.incomplete_candidate_count_groups;
                break;
            }
        }
        diagnostics.retained_candidates += retained.size();
        return;
    }
    for (size_t pos : group.route_positions) advance(pos, 0);
    observe_stage(CandidateStage::Rank);
    const auto later = [](const CandidateMergeCursor& a, const CandidateMergeCursor& b) {
        return candidate_merge_key_less(b.candidate, a.candidate);
    };
    std::make_heap(scratch.heads.begin(), scratch.heads.end(), later);
    while (!scratch.heads.empty()) {
        std::pop_heap(scratch.heads.begin(), scratch.heads.end(), later);
        const auto head = scratch.heads.back();
        scratch.heads.pop_back();
        // The first occurrence has the best score/rank/route provenance,
        // including ties at the shortlist boundary.
        if (observe_hash([&] { return scratch.insert(head.candidate.sample_id); })) {
            retained.push_back(head.candidate);
            ++diagnostics.unique_candidates;
        }
        if (retained.size() == limit) {
            // Do not pretend the unvisited tail was counted/deduplicated.
            ++diagnostics.incomplete_candidate_count_groups;
            break;
        }
        const size_t old_size = scratch.heads.size();
        advance(head.route_position, head.candidate.source_rank + 1);
        if (scratch.heads.size() != old_size)
            std::push_heap(scratch.heads.begin(), scratch.heads.end(), later);
    }
    diagnostics.retained_candidates += retained.size();
    return;
}

template <typename RouteFilter, typename StageObserver = IgnoreCandidateStage,
          typename HashObserver = DirectCandidateHash>
std::vector<Candidate> merge_sorted_group_candidates(
    const RouteGroup& group, const std::vector<uint32_t>& route_ids,
    const std::vector<CandidateId>& ids, const std::vector<float>& distances,
    uint32_t width, uint64_t base_rows, uint32_t cap, RouteFilter&& filter,
    Diagnostics& diagnostics, StageObserver observe_stage = {}, HashObserver observe_hash = {})
{
    std::vector<Candidate> retained;
    merge_sorted_group_candidates_into(group, route_ids, ids, distances, width, base_rows,
        cap, std::forward<RouteFilter>(filter), diagnostics, retained, observe_stage, observe_hash);
    return retained;
}

template <typename ResultId>
void write_group_result_to_representative_route(
    const RouteGroup& group,
    const std::vector<ResultId>& group_result,
    const std::vector<float>& group_distances,
    int group_step,
    std::vector<std::vector<ResultId>>& route_results,
    std::vector<int>& route_steps,
    std::vector<std::vector<float>>& route_distances,
    Diagnostics& diagnostics)
{
    if (group_result.size() != group_distances.size()) {
        throw std::runtime_error("low-frequency merge result and distance widths differ");
    }
    for (size_t route_pos : group.route_positions) {
        if (route_pos >= route_results.size() ||
            route_pos >= route_steps.size() ||
            route_pos >= route_distances.size()) {
            throw std::runtime_error("low-frequency merge output route position is out of range");
        }
    }
    for (size_t route_pos : group.route_positions) {
        route_steps[route_pos] = group_step;
    }
    diagnostics.route_step_rows += group.route_positions.size();
    if (group.route_positions.empty() || group_result.empty()) {
        diagnostics.compact_empty_result_rows += group.route_positions.size();
        return;
    }

    const size_t representative_route_pos = group.route_positions.front();
    route_results[representative_route_pos] = group_result;
    route_distances[representative_route_pos] = group_distances;
    ++diagnostics.representative_result_rows;
    diagnostics.compact_empty_result_rows += group.route_positions.size() - 1;
}

template <typename ResultId>
bool route_result_row_is_sorted(
    size_t route_idx,
    const std::vector<std::vector<ResultId>>& route_results,
    const std::vector<std::vector<float>>& route_distances)
{
    bool saw_invalid = false;
    bool have_previous = false;
    float previous_distance = std::numeric_limits<float>::max();
    ResultId previous_id = static_cast<ResultId>(UINT32_MAX);
    for (size_t rank = 0; rank < route_results[route_idx].size(); ++rank) {
        const ResultId sample_id = route_results[route_idx][rank];
        if (sample_id == static_cast<ResultId>(UINT32_MAX)) {
            saw_invalid = true;
            continue;
        }
        if (saw_invalid || route_idx >= route_distances.size() ||
            rank >= route_distances[route_idx].size()) {
            return false;
        }
        const float distance = route_distances[route_idx][rank];
        const bool precedes_previous =
            have_previous &&
            (distance < previous_distance ||
             (distance == previous_distance && sample_id < previous_id));
        if (std::isnan(distance) || precedes_previous) {
            return false;
        }
        previous_distance = distance;
        previous_id = sample_id;
        have_previous = true;
    }
    return true;
}

template <typename ResultId>
struct RouteMergeNode {
    float distance = std::numeric_limits<float>::max();
    ResultId sample_id = static_cast<ResultId>(UINT32_MAX);
    size_t route_idx = 0;
    size_t rank = 0;
};

template <typename ResultId>
struct RouteMergeNodeAfter {
    bool operator()(const RouteMergeNode<ResultId>& lhs,
                    const RouteMergeNode<ResultId>& rhs) const
    {
        if (lhs.distance != rhs.distance) {
            return lhs.distance > rhs.distance;
        }
        if (lhs.sample_id != rhs.sample_id) {
            return lhs.sample_id > rhs.sample_id;
        }
        if (lhs.route_idx != rhs.route_idx) {
            return lhs.route_idx > rhs.route_idx;
        }
        return lhs.rank > rhs.rank;
    }
};

template <typename ResultId>
bool try_merge_single_route_fast_path(
    const std::vector<uint32_t>& route_to_original_query,
    uint32_t original_query_count,
    const std::vector<std::vector<ResultId>>& route_results,
    const std::vector<std::vector<float>>& route_distances,
    const std::vector<int>& route_steps,
    uint32_t topk,
    std::vector<std::vector<ResultId>>& merged_results,
    std::vector<int>& merged_steps,
    std::vector<std::vector<float>>* merged_distances)
{
    if (route_results.size() != route_steps.size() ||
        route_to_original_query.size() < route_results.size()) {
        return false;
    }
    std::vector<uint8_t> route_seen(original_query_count, 0);
    for (size_t route_idx = 0; route_idx < route_results.size(); ++route_idx) {
        const uint32_t original_query = route_to_original_query[route_idx];
        if (original_query >= original_query_count) {
            continue;
        }
        if (route_seen[original_query] != 0) {
            return false;
        }
        route_seen[original_query] = 1;
        if (!route_result_row_is_sorted(route_idx, route_results, route_distances)) {
            return false;
        }
    }
    merged_results.assign(
        original_query_count,
        std::vector<ResultId>(topk, static_cast<ResultId>(UINT32_MAX)));
    merged_steps.assign(original_query_count, 0);
    if (merged_distances != nullptr) {
        merged_distances->assign(
            original_query_count,
            std::vector<float>(topk, std::numeric_limits<float>::max()));
    }
    for (size_t route_idx = 0; route_idx < route_results.size(); ++route_idx) {
        const uint32_t original_query = route_to_original_query[route_idx];
        if (original_query >= original_query_count) {
            continue;
        }
        merged_steps[original_query] = route_steps[route_idx];
        size_t output_rank = 0;
        for (size_t rank = 0;
             rank < route_results[route_idx].size() && output_rank < topk;
             ++rank) {
            const ResultId sample_id = route_results[route_idx][rank];
            if (sample_id == static_cast<ResultId>(UINT32_MAX)) {
                break;
            }
            merged_results[original_query][output_rank] = sample_id;
            if (merged_distances != nullptr) {
                (*merged_distances)[original_query][output_rank] =
                    route_distances[route_idx][rank];
            }
            ++output_rank;
        }
    }
    return true;
}

template <typename ResultId>
bool try_merge_sorted_routes_fast_path(
    const std::vector<uint32_t>& route_to_original_query,
    uint32_t original_query_count,
    const std::vector<std::vector<ResultId>>& route_results,
    const std::vector<std::vector<float>>& route_distances,
    const std::vector<int>& route_steps,
    uint32_t topk,
    std::vector<std::vector<ResultId>>& merged_results,
    std::vector<int>& merged_steps,
    std::vector<std::vector<float>>* merged_distances)
{
    std::vector<size_t> route_offsets(static_cast<size_t>(original_query_count) + 1, 0);
    std::vector<int64_t> step_sums(original_query_count, 0);
    for (size_t route_idx = 0; route_idx < route_results.size(); ++route_idx) {
        if (route_idx >= route_to_original_query.size()) {
            continue;
        }
        const uint32_t original_query = route_to_original_query[route_idx];
        if (original_query >= original_query_count) {
            continue;
        }
        if (!route_result_row_is_sorted(route_idx, route_results, route_distances)) {
            return false;
        }
        ++route_offsets[static_cast<size_t>(original_query) + 1];
        if (route_idx < route_steps.size()) {
            step_sums[original_query] += static_cast<int64_t>(route_steps[route_idx]);
            if (step_sums[original_query] < std::numeric_limits<int>::min() ||
                step_sums[original_query] > std::numeric_limits<int>::max()) {
                throw std::runtime_error("route step aggregation overflows int");
            }
        }
    }
    for (size_t original_query = 0; original_query < original_query_count; ++original_query) {
        route_offsets[original_query + 1] += route_offsets[original_query];
    }
    std::vector<size_t> route_indices(route_offsets.back());
    std::vector<size_t> route_cursors = route_offsets;
    for (size_t route_idx = 0; route_idx < route_results.size(); ++route_idx) {
        if (route_idx >= route_to_original_query.size()) {
            continue;
        }
        const uint32_t original_query = route_to_original_query[route_idx];
        if (original_query >= original_query_count) {
            continue;
        }
        route_indices[route_cursors[original_query]++] = route_idx;
    }

    merged_results.assign(
        original_query_count,
        std::vector<ResultId>(topk, static_cast<ResultId>(UINT32_MAX)));
    merged_steps.resize(original_query_count);
    for (size_t original_query = 0; original_query < original_query_count; ++original_query) {
        merged_steps[original_query] = static_cast<int>(step_sums[original_query]);
    }
    if (merged_distances != nullptr) {
        merged_distances->assign(
            original_query_count,
            std::vector<float>(topk, std::numeric_limits<float>::max()));
    }

    std::vector<RouteMergeNode<ResultId>> heap;
    RouteMergeNodeAfter<ResultId> node_after;
    for (size_t original_query = 0; original_query < original_query_count;
         ++original_query) {
        heap.clear();
        const size_t route_begin = route_offsets[original_query];
        const size_t route_end = route_offsets[original_query + 1];
        if (heap.capacity() < route_end - route_begin) {
            heap.reserve(route_end - route_begin);
        }
        for (size_t route_pos = route_begin; route_pos < route_end; ++route_pos) {
            const size_t route_idx = route_indices[route_pos];
            if (route_results[route_idx].empty() ||
                route_results[route_idx].front() == static_cast<ResultId>(UINT32_MAX)) {
                continue;
            }
            heap.push_back({route_distances[route_idx].front(),
                            route_results[route_idx].front(),
                            route_idx,
                            0});
        }
        std::make_heap(heap.begin(), heap.end(), node_after);

        size_t output_rank = 0;
        while (!heap.empty() && output_rank < topk) {
            std::pop_heap(heap.begin(), heap.end(), node_after);
            const RouteMergeNode<ResultId> node = heap.back();
            heap.pop_back();

            const size_t next_rank = node.rank + 1;
            if (next_rank < route_results[node.route_idx].size()) {
                const ResultId next_id = route_results[node.route_idx][next_rank];
                if (next_id != static_cast<ResultId>(UINT32_MAX)) {
                    heap.push_back({route_distances[node.route_idx][next_rank],
                                    next_id,
                                    node.route_idx,
                                    next_rank});
                    std::push_heap(heap.begin(), heap.end(), node_after);
                }
            }

            const auto output_begin = merged_results[original_query].begin();
            if (std::find(output_begin, output_begin + output_rank, node.sample_id) !=
                output_begin + output_rank) {
                continue;
            }
            merged_results[original_query][output_rank] = node.sample_id;
            if (merged_distances != nullptr) {
                (*merged_distances)[original_query][output_rank] = node.distance;
            }
            ++output_rank;
        }
    }
    return true;
}

template <typename ResultId>
FinalMergeStrategy merge_route_results_to_original(
    const std::vector<uint32_t>& route_to_original_query,
    uint32_t original_query_count,
    const std::vector<std::vector<ResultId>>& route_results,
    const std::vector<std::vector<float>>& route_distances,
    const std::vector<int>& route_steps,
    uint32_t topk,
    std::vector<std::vector<ResultId>>& merged_results,
    std::vector<int>& merged_steps,
    std::vector<std::vector<float>>* merged_distances = nullptr)
{
    if (try_merge_single_route_fast_path(route_to_original_query,
                                         original_query_count,
                                         route_results,
                                         route_distances,
                                         route_steps,
                                         topk,
                                         merged_results,
                                         merged_steps,
                                         merged_distances)) {
        return FinalMergeStrategy::SingleRouteCopy;
    }
    if (try_merge_sorted_routes_fast_path(route_to_original_query,
                                          original_query_count,
                                          route_results,
                                          route_distances,
                                          route_steps,
                                          topk,
                                          merged_results,
                                          merged_steps,
                                          merged_distances)) {
        return FinalMergeStrategy::SortedRouteHeap;
    }
    merged_results.assign(
        original_query_count,
        std::vector<ResultId>(topk, static_cast<ResultId>(UINT32_MAX)));
    merged_steps.assign(original_query_count, 0);
    if (merged_distances != nullptr) {
        merged_distances->assign(
            original_query_count,
            std::vector<float>(topk, std::numeric_limits<float>::max()));
    }
    std::vector<std::unordered_map<ResultId, float>> candidates(original_query_count);

    for (size_t route_idx = 0; route_idx < route_results.size(); ++route_idx) {
        if (route_idx >= route_to_original_query.size()) {
            continue;
        }
        const uint32_t original_query = route_to_original_query[route_idx];
        if (original_query >= original_query_count) {
            continue;
        }
        if (route_idx < route_steps.size()) {
            const int64_t merged_step =
                static_cast<int64_t>(merged_steps[original_query]) +
                static_cast<int64_t>(route_steps[route_idx]);
            if (merged_step < std::numeric_limits<int>::min() ||
                merged_step > std::numeric_limits<int>::max()) {
                throw std::runtime_error("route step aggregation overflows int");
            }
            merged_steps[original_query] = static_cast<int>(merged_step);
        }

        auto& candidate_map = candidates[original_query];
        for (size_t rank = 0; rank < route_results[route_idx].size(); ++rank) {
            const ResultId sample_id = route_results[route_idx][rank];
            if (sample_id == static_cast<ResultId>(UINT32_MAX)) {
                continue;
            }
            const float distance =
                route_idx < route_distances.size() && rank < route_distances[route_idx].size()
                ? route_distances[route_idx][rank]
                : std::numeric_limits<float>::max();
            auto it = candidate_map.find(sample_id);
            if (it == candidate_map.end() || distance < it->second) {
                candidate_map[sample_id] = distance;
            }
        }
    }

    for (uint32_t original_query = 0; original_query < original_query_count;
         ++original_query) {
        const auto& candidate_map = candidates[original_query];
        if (candidate_map.empty()) {
            continue;
        }
        std::vector<std::pair<float, ResultId>> scored;
        scored.reserve(candidate_map.size());
        for (const auto& entry : candidate_map) {
            scored.emplace_back(entry.second, entry.first);
        }
        std::sort(scored.begin(), scored.end(), [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) {
                return lhs.first < rhs.first;
            }
            return lhs.second < rhs.second;
        });
        for (size_t rank = 0; rank < scored.size() && rank < topk; ++rank) {
            merged_results[original_query][rank] = scored[rank].second;
            if (merged_distances != nullptr) {
                (*merged_distances)[original_query][rank] = scored[rank].first;
            }
        }
    }
    return FinalMergeStrategy::HashDedupSort;
}

}  // namespace go_fann::backend_rabitq::lowfreq_merge
