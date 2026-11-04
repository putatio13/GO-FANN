#include "execution/posting_segment_plan.h"
#include "search/common/posting_shortlist_heap.cuh"

#include <cassert>
#include <cfloat>
#include <random>

int main()
{
    using namespace go_fann;
    const std::vector<uint32_t> routes{7, 3, 11, 5};
    const std::vector<uint64_t> offsets{0, 100, 1000, 30000};
    const std::vector<uint64_t> sizes{0, 31, 20001, 8};
    const auto plan = posting_segments::make_plan(routes, offsets, sizes, 1024);
    for (size_t q = 0; q < routes.size(); ++q) {
        uint64_t cursor = offsets[q], shortest = UINT64_MAX, longest = 0;
        assert(plan.query_indptr[q + 1] > plan.query_indptr[q]);
        for (size_t p = plan.query_indptr[q]; p < plan.query_indptr[q + 1]; ++p) {
            assert(plan.route_ids[p] == routes[q]);
            assert(plan.offsets[p] == cursor);
            assert(plan.sizes[p] <= 1024);
            cursor += plan.sizes[p];
            shortest = std::min(shortest, plan.sizes[p]);
            longest = std::max(longest, plan.sizes[p]);
        }
        assert(cursor == offsets[q] + sizes[q]);
        assert(longest - shortest <= 1);
    }
    bool rejected = false;
    try { posting_segments::make_plan({1}, {UINT64_MAX}, {1}, 32); }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { posting_segments::make_plan({1}, {0}, {1}, 0); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { posting_segments::make_plan({1}, {0}, {UINT64_MAX}, 1); }
    catch (const std::overflow_error&) { rejected = true; }
    assert(rejected);

    // Ties, short/empty slices and K larger than a segment exercise padding.
    std::mt19937 random(7123);
    for (int k : {10, 100, 400, 4096}) {
        std::vector<float> scores(21007);
        for (auto& score : scores) score = static_cast<float>(random() % 400);
        for (uint32_t target : {32U, 1024U, 4096U}) {
            for (uint64_t rows : {0ULL, 17ULL, 21007ULL}) {
                const auto segments = posting_segments::make_plan({0}, {0}, {rows}, target);
                std::vector<float> full_dist(k, FLT_MAX), merged_dist(k, FLT_MAX);
                std::vector<uint32_t> full(k, UINT32_MAX), merged(k, UINT32_MAX);
                for (uint32_t id = 0; id < rows; ++id)
                    posting_shortlist_heap::insert(scores[id], id, full_dist.data(), full.data(), k);
                for (size_t part = 0; part < segments.sizes.size(); ++part) {
                    std::vector<float> dist(k, FLT_MAX);
                    std::vector<uint32_t> ids(k, UINT32_MAX);
                    const uint64_t end = segments.offsets[part] + segments.sizes[part];
                    for (uint32_t id = segments.offsets[part]; id < end; ++id)
                        posting_shortlist_heap::insert(scores[id], id, dist.data(), ids.data(), k);
                    for (int i = 0; i < k; ++i)
                        posting_shortlist_heap::insert(dist[i], ids[i], merged_dist.data(), merged.data(), k);
                }
                posting_shortlist_heap::sort(full_dist.data(), full.data(), k);
                posting_shortlist_heap::sort(merged_dist.data(), merged.data(), k);
                assert(full == merged && full_dist == merged_dist);
            }
        }
    }
}
