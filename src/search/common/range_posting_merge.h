#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace go_fann::range_posting {

template <typename Id>
void merge_exact_result_row(std::vector<Id>& target_ids,
                            std::vector<float>& target_distances,
                            const std::vector<Id>& source_ids,
                            const std::vector<float>& source_distances,
                            uint32_t topk)
{
    std::unordered_map<uint32_t, float> best_by_sample;
    best_by_sample.reserve(target_ids.size() + source_ids.size());
    auto collect = [&best_by_sample](const std::vector<Id>& ids,
                                     const std::vector<float>& distances) {
        const size_t count = std::min(ids.size(), distances.size());
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] == static_cast<Id>(UINT32_MAX) ||
                ids[i] > static_cast<Id>(UINT32_MAX) ||
                !std::isfinite(distances[i])) {
                continue;
            }
            const uint32_t sample_id = static_cast<uint32_t>(ids[i]);
            auto [it, inserted] = best_by_sample.emplace(sample_id, distances[i]);
            if (!inserted && distances[i] < it->second) {
                it->second = distances[i];
            }
        }
    };
    collect(target_ids, target_distances);
    collect(source_ids, source_distances);

    std::vector<std::pair<float, uint32_t>> sorted;
    sorted.reserve(best_by_sample.size());
    for (const auto& [sample_id, distance] : best_by_sample) {
        sorted.emplace_back(distance, sample_id);
    }
    std::sort(sorted.begin(), sorted.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first < rhs.first ||
            (lhs.first == rhs.first && lhs.second < rhs.second);
    });

    target_ids.assign(topk, static_cast<Id>(UINT32_MAX));
    target_distances.assign(topk, FLT_MAX);
    for (size_t i = 0; i < sorted.size() && i < topk; ++i) {
        target_distances[i] = sorted[i].first;
        target_ids[i] = static_cast<Id>(sorted[i].second);
    }
}

}  // namespace go_fann::range_posting
