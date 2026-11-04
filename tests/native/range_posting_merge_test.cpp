#include "search/common/range_posting_merge.h"

#include <cassert>
#include <cfloat>
#include <cstdint>
#include <vector>

namespace rp = go_fann::range_posting;

int main()
{
    using Id = uint64_t;
    const Id invalid = UINT32_MAX;

    // mode=off does not invoke the merge helper and preserves graph output.
    const std::vector<Id> graph_ids = {7, 8, invalid};
    const std::vector<float> graph_distances = {1.0f, 3.0f, FLT_MAX};
    auto off_ids = graph_ids;
    auto off_distances = graph_distances;
    assert(off_ids == graph_ids && off_distances == graph_distances);

    // mode=select starts from an empty graph row and returns posting top-k.
    std::vector<Id> select_ids;
    std::vector<float> select_distances;
    rp::merge_exact_result_row(select_ids,
                               select_distances,
                               std::vector<Id>{11, 10, 12},
                               std::vector<float>{2.0f, 1.0f, 3.0f},
                               2);
    assert((select_ids == std::vector<Id>{10, 11}));
    assert((select_distances == std::vector<float>{1.0f, 2.0f}));

    // mode=hybrid unions graph and posting rows, deduplicates, and keeps exact top-k.
    auto hybrid_ids = graph_ids;
    auto hybrid_distances = graph_distances;
    rp::merge_exact_result_row(hybrid_ids,
                               hybrid_distances,
                               std::vector<Id>{8, 9, 10, 9},
                               std::vector<float>{2.0f, 0.5f, 4.0f, 0.75f},
                               3);
    assert((hybrid_ids == std::vector<Id>{9, 7, 8}));
    assert((hybrid_distances == std::vector<float>{0.5f, 1.0f, 2.0f}));
    return 0;
}
