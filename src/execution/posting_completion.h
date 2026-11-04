#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <thread>
#include <vector>

namespace go_fann::posting_completion {

struct Groups {
    std::vector<std::vector<size_t>> positions;
    std::vector<size_t> route_to_group;

    static Groups make(const std::vector<uint32_t>& original_ids, size_t original_count)
    {
        Groups groups;
        std::vector<size_t> by_original(original_count, SIZE_MAX);
        for (size_t route = 0; route < original_ids.size(); ++route) {
            const uint32_t original = original_ids[route];
            if (original >= original_count) throw std::out_of_range("posting original query ID");
            if (by_original[original] == SIZE_MAX) {
                by_original[original] = groups.positions.size();
                groups.positions.emplace_back();
            }
            const size_t group = by_original[original];
            groups.route_to_group.push_back(group);
            groups.positions[group].push_back(route);
        }
        return groups;
    }
};

// Consume a group only once every route is complete. A large group may exceed
// the batching hint; splitting it would change deduplication and rerank budgets.
template <typename Ready, typename Consume>
void consume(const Groups& groups, size_t batch_hint, const std::atomic<bool>& cancelled,
             Ready ready, Consume consume_batch)
{
    if (batch_hint == 0) throw std::invalid_argument("posting batch hint must be positive");
    std::vector<size_t> remaining;
    for (const auto& positions : groups.positions) remaining.push_back(positions.size());
    std::vector<uint8_t> seen(groups.route_to_group.size(), 0);
    std::deque<size_t> completed;
    size_t consumed = 0;
    std::vector<size_t> batch;
    batch.reserve(batch_hint);
    while (consumed < seen.size()) {
        if (cancelled.load(std::memory_order_acquire)) return;
        for (size_t route = 0; route < seen.size(); ++route) {
            if (seen[route] || !ready(route)) continue;
            seen[route] = 1;
            const size_t group = groups.route_to_group[route];
            if (--remaining[group] == 0) completed.push_back(group);
        }
        if (completed.empty()) {
            std::this_thread::sleep_for(std::chrono::microseconds(2));
            continue;
        }
        batch.clear();
        while (!completed.empty()) {
            const auto& positions = groups.positions[completed.front()];
            if (!batch.empty() && batch.size() + positions.size() > batch_hint) break;
            batch.insert(batch.end(), positions.begin(), positions.end());
            completed.pop_front();
            if (batch.size() >= batch_hint) break;
        }
        consume_batch(batch);
        consumed += batch.size();
    }
}

}  // namespace go_fann::posting_completion
