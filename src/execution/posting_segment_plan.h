#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace go_fann::posting_segments {

struct Plan {
    std::vector<uint32_t> route_ids;
    std::vector<uint32_t> query_positions;
    std::vector<uint64_t> offsets;
    std::vector<uint64_t> sizes;
    std::vector<uint32_t> query_indptr{0};
};

inline Plan make_plan(const std::vector<uint32_t>& route_ids,
                      const std::vector<uint64_t>& offsets,
                      const std::vector<uint64_t>& sizes,
                      uint32_t target_rows)
{
    if (target_rows == 0 || route_ids.size() != offsets.size() ||
        offsets.size() != sizes.size()) {
        throw std::invalid_argument("invalid posting segment plan inputs");
    }
    Plan plan;
    for (size_t query = 0; query < route_ids.size(); ++query) {
        const uint64_t rows = sizes[query];
        if (offsets[query] > std::numeric_limits<uint64_t>::max() - rows) {
            throw std::overflow_error("posting segment offset overflow");
        }
        // Keep one empty segment so every query still produces a padded row.
        const uint64_t count = rows == 0 ? 1 : 1 + (rows - 1) / target_rows;
        if (count > UINT32_MAX - plan.route_ids.size()) {
            throw std::overflow_error("posting segment count exceeds uint32 range");
        }
        const uint64_t width = rows / count;
        const uint64_t remainder = rows % count;
        for (uint64_t part = 0; part < count; ++part) {
            plan.route_ids.push_back(route_ids[query]);
            plan.query_positions.push_back(static_cast<uint32_t>(query));
            plan.offsets.push_back(offsets[query] + part * width +
                                   std::min(part, remainder));
            plan.sizes.push_back(width + (part < remainder ? 1 : 0));
        }
        plan.query_indptr.push_back(static_cast<uint32_t>(plan.route_ids.size()));
    }
    return plan;
}

}  // namespace go_fann::posting_segments
