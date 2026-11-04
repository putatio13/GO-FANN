#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

namespace go_fann::pq_posting_streaming {

// Positions are compact scan rows. Route IDs remain caller-owned and may be
// arbitrarily permuted. Consume returns only once its workers have drained.
template<class Ready, class Consume>
void consume_ready(size_t rows, size_t batch_size, Ready&& ready, Consume&& consume,
                   const std::atomic<bool>& cancelled) {
    if (!batch_size) throw std::invalid_argument("posting stream batch size is zero");
    std::vector<uint8_t> consumed(rows, 0);
    std::vector<size_t> positions;
    positions.reserve(batch_size < rows ? batch_size : rows);
    size_t completed = 0;
    while (completed < rows) {
        if (cancelled.load(std::memory_order_acquire)) return;
        positions.clear();
        for (size_t pos = 0; pos < rows && positions.size() < batch_size; ++pos)
            if (!consumed[pos] && ready(pos)) positions.push_back(pos);
        if (positions.empty()) {
            std::this_thread::yield();
            continue;
        }
        consume(positions);
        for (size_t pos : positions) { consumed[pos] = 1; ++completed; }
    }
}

inline bool predicate_verified(const std::vector<uint8_t>* coverage, uint32_t route) {
    return coverage && route < coverage->size() && (*coverage)[route] != 0;
}

} // namespace go_fann::pq_posting_streaming
