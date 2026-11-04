#include "search/common/posting_shortlist_heap.cuh"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

using Id = uint64_t;
using Pair = std::pair<float, Id>;

void verify_top_m(const std::vector<Pair>& candidates, int shortlist_k)
{
    std::vector<float> distances(static_cast<size_t>(shortlist_k), FLT_MAX);
    std::vector<Id> ids(static_cast<size_t>(shortlist_k), UINT32_MAX);
    for (const auto& [distance, id] : candidates) {
        go_fann::posting_shortlist_heap::insert(
            distance, id, distances.data(), ids.data(), shortlist_k);
    }
    go_fann::posting_shortlist_heap::sort(
        distances.data(), ids.data(), shortlist_k);

    std::vector<Pair> expected;
    expected.reserve(candidates.size());
    for (const Pair& candidate : candidates) {
        if (candidate.second != UINT32_MAX) expected.push_back(candidate);
    }
    std::sort(expected.begin(), expected.end());
    expected.resize(std::min(expected.size(), static_cast<size_t>(shortlist_k)));

    for (size_t i = 0; i < expected.size(); ++i) {
        assert(distances[i] == expected[i].first);
        assert(ids[i] == expected[i].second);
    }
    for (size_t i = expected.size(); i < distances.size(); ++i) {
        assert(distances[i] == FLT_MAX);
        assert(ids[i] == UINT32_MAX);
    }
}

}  // namespace

int main()
{
    namespace heap = go_fann::posting_shortlist_heap;
    // can_enter is safe for empty capacity/null storage and preserves sentinel,
    // ties, infinities and NaN behavior of the original insert comparator.
    assert(!heap::can_enter(0.0f, Id(1), nullptr, static_cast<const Id*>(nullptr), 0));
    float d[] = {FLT_MAX, FLT_MAX}; Id ids[] = {UINT32_MAX, UINT32_MAX};
    assert(heap::can_enter(FLT_MAX, Id(4), d, ids, 2));
    assert(!heap::can_enter(0.0f, Id(UINT32_MAX), d, ids, 2));
    assert(!heap::can_enter(std::numeric_limits<float>::infinity(), Id(1), d, ids, 2));
    assert(!heap::can_enter(std::numeric_limits<float>::quiet_NaN(), Id(1), d, ids, 2));
    heap::insert(2.0f, Id(9), d, ids, 2);
    assert(heap::can_enter(100.0f, Id(1), d, ids, 2)); // Unfilled heap.
    heap::insert(2.0f, Id(7), d, ids, 2);
    assert(heap::can_enter(2.0f, Id(8), d, ids, 2));
    assert(!heap::can_enter(2.0f, Id(9), d, ids, 2));
    assert(!heap::can_enter(2.0f, Id(10), d, ids, 2));
    assert(heap::can_enter(-std::numeric_limits<float>::infinity(), Id(1), d, ids, 2));
    for (float score : {1.0f, 2.0f, 3.0f, FLT_MAX,
                        std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
        for (Id id : {Id(1), Id(9), Id(10), Id(UINT32_MAX)}) {
            float copy_d[] = {d[0], d[1]}; Id copy_ids[] = {ids[0], ids[1]};
            const bool admitted = heap::can_enter(score, id, copy_d, copy_ids, 2);
            heap::insert(score, id, copy_d, copy_ids, 2);
            const bool changed = copy_d[0] != d[0] || copy_d[1] != d[1] ||
                                 copy_ids[0] != ids[0] || copy_ids[1] != ids[1];
            assert(admitted == changed);
        }
    }
    verify_top_m({{9.0f, 9}, {1.0f, 7}, {4.0f, 4}, {1.0f, 3},
                  {2.0f, 8}, {0.5f, UINT32_MAX}, {3.0f, 2}},
                 4);
    verify_top_m({{3.0f, 4}, {2.0f, 5}, {1.0f, 6}}, 1);

    std::vector<Pair> large;
    large.reserve(10000);
    for (uint64_t i = 0; i < 10000; ++i) {
        large.emplace_back(static_cast<float>((i * 37) % 997), 10000 - i);
    }
    verify_top_m(large, 4000);
    return 0;
}
