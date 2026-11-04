#include "search/rabitq/rabitq_lowfreq_merge.h"
#include <chrono>
#include <iostream>
#include <random>
#include <stdexcept>

namespace merge = go_fann::backend_rabitq::lowfreq_merge;

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

void compare(const merge::RouteGroup& group, const std::vector<uint32_t>& routes,
             const std::vector<merge::CandidateId>& ids, const std::vector<float>& scores,
             uint32_t width, uint32_t cap, bool filtered)
{
    const auto filter = [filtered](uint32_t route, uint32_t sample) {
        return !filtered || (sample + route) % 7 != 0;
    };
    merge::Diagnostics before, after;
    const auto a = merge::select_group_candidates(group, routes, ids, scores, width, 100000,
                                                   cap, filter, before);
    const auto b = merge::merge_sorted_group_candidates(group, routes, ids, scores, width, 100000,
                                                        cap, filter, after);
    thread_local std::vector<merge::Candidate> reused;
    merge::Diagnostics into_stats;
    merge::merge_sorted_group_candidates_into(group, routes, ids, scores, width, 100000,
                                               cap, filter, into_stats, reused);
    require(reused.size() == b.size(), "reused output has stale candidates");
    for (size_t i = 0; i < b.size(); ++i)
        require(reused[i].sample_id == b[i].sample_id &&
                reused[i].approximate_distance == b[i].approximate_distance &&
                reused[i].source_rank == b[i].source_rank &&
                reused[i].source_route_id == b[i].source_route_id,
                "reusable selector changed candidate provenance");
    require(a.size() == b.size(), "retained size differs from legacy oracle");
    for (size_t i = 0; i < a.size(); ++i) {
        require(a[i].sample_id == b[i].sample_id &&
                a[i].approximate_distance == b[i].approximate_distance &&
                a[i].source_rank == b[i].source_rank &&
                a[i].source_route_id == b[i].source_route_id,
                "candidate order, score or duplicate provenance changed");
    }
    require(after.route_candidate_slots == before.route_candidate_slots,
            "input slot accounting changed");
    require(after.retained_candidates == before.retained_candidates, "retained accounting changed");
    if (!after.incomplete_candidate_count_groups) {
        require(after.route_valid_candidates == before.route_valid_candidates &&
                after.unique_candidates == before.unique_candidates, "complete counts differ");
    }
}

void randomized_oracle()
{
    std::mt19937 random(20261017);
    for (unsigned trial = 0; trial != 5000; ++trial) {
        const uint32_t count = random() % 65;
        const uint32_t width = random() % 129;
        const uint32_t cap = random() % 260;
        merge::RouteGroup group{0, {}};
        std::vector<uint32_t> routes(count);
        std::vector<merge::CandidateId> ids;
        std::vector<float> scores;
        for (uint32_t route = 0; route < count; ++route) {
            routes[route] = count - route;
            group.route_positions.push_back(route);
            std::vector<std::pair<float, uint32_t>> row;
            for (uint32_t rank = 0; rank < width; ++rank) {
                const uint32_t sample = random() % 40; // heavy duplicates, including within a route
                row.emplace_back(float(random() % 10), sample);
            }
            std::sort(row.begin(), row.end());
            if (trial % 3 == 0) std::shuffle(row.begin(), row.end(), random);
            for (const auto& item : row) {
                const auto invalid = random() % 20;
                ids.push_back(invalid == 0 ? UINT32_MAX : invalid == 1 ? 100001 : item.second);
                scores.push_back(item.first);
            }
        }
        std::shuffle(group.route_positions.begin(), group.route_positions.end(), random);
        compare(group, routes, ids, scores, width, cap, trial % 2);
    }
}

void edge_cases()
{
    // Same ID with different scores and route-local rejection; equal-distance
    // boundary ties must use ID, then rank/route for provenance only.
    compare({0, {1, 0, 2}}, {8, 3, 1},
            {9, 5, 8, 9, 3, 5, 5, 9, UINT32_MAX},
            {0, 1, 2, 0, 1, 1, 1, 2, 0}, 3, 3, true);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    compare({0, {0, 1}}, {0, 1}, {9, 2, 3, 1, 2, 4}, {nan, inf, -inf, -0.f, 0.f, 1.f}, 3, 6, false);
    compare({0, {0}}, {0}, {UINT32_MAX, UINT32_MAX}, {nan, nan}, 2, 100, false);
    bool failed = false;
    try {
        merge::Diagnostics stats;
        (void)merge::merge_sorted_group_candidates({0, {2}}, {0}, {1}, {0.f}, 1, 10, 1,
                                                   [](auto, auto) { return true; }, stats);
    } catch (const std::runtime_error&) { failed = true; }
    require(failed, "bad route mapping was not rejected");
}

void bounded_merge_work()
{
    merge::RouteGroup group{0, {}};
    std::vector<uint32_t> routes;
    std::vector<merge::CandidateId> ids;
    std::vector<float> scores;
    for (uint32_t r = 0; r < 32; ++r) {
        routes.push_back(r); group.route_positions.push_back(r);
        for (uint32_t i = 0; i < 100; ++i) {
            ids.push_back(i * 32 + r); scores.push_back(float(i * 32 + r));
        }
    }
    merge::Diagnostics stats;
    const auto result = merge::merge_sorted_group_candidates(group, routes, ids, scores, 100, 100000,
        100, [](auto, auto) { return true; }, stats);
    require(result.size() == 100 && result.back().sample_id == 99, "large-route merge failed");
    require(stats.sorted_merge_groups == 1 && stats.fallback_merge_groups == 0 &&
            stats.merge_visited_slots < 150 && stats.incomplete_candidate_count_groups == 1,
            "sorted path scanned the complete union after reaching cap");
}

int main()
{
    edge_cases();
    randomized_oracle();
    bounded_merge_work();
    std::cout << "candidate merge: 5000 oracle cases and edge/bounded-work cases passed\n";
}
