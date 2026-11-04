#include "search/pq/pq_posting_streaming.h"
#include "execution/rabitq_cpu_worker_pool.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <future>
#include <iostream>
#include <numeric>

using go_fann::pq_posting_streaming::consume_ready;
using go_fann::pq_posting_streaming::predicate_verified;
using go_fann::backend_rabitq::RabitqCpuWorkerPool;

int main() {
    constexpr size_t rows = 7, width = 5;
    const std::array<uint32_t, rows> routes{41, 2, 90, 17, 4, 66, 1};
    std::array<std::array<uint32_t, width>, rows> mapped{};
    std::array<std::atomic<bool>, rows> ready{};
    std::array<std::vector<uint32_t>, rows> result;
    std::array<unsigned, rows> seen{};
    std::atomic<bool> cancelled{false};
    std::promise<void> first_consumed;
    auto first = first_consumed.get_future();
    RabitqCpuWorkerPool pool(3);
    auto consumer = std::async(std::launch::async, [&] {
        consume_ready(rows, 2, [&](size_t p) { return ready[p].load(std::memory_order_acquire); },
            [&](const std::vector<size_t>& positions) {
                pool.submit(positions.size(), [&](size_t i) {
                    const size_t p = positions[i];
                    // The fixture's row order is intentionally significant.
                    assert(mapped[p][0] == routes[p] * 10);
                    result[p].assign(mapped[p].begin(), mapped[p].end());
                    ++seen[p];
                })->wait();
                if (positions.front() == 5) first_consumed.set_value();
            }, cancelled);
    });
    auto publish = [&](size_t p) {
        std::iota(mapped[p].begin(), mapped[p].end(), routes[p] * 10);
        ready[p].store(true, std::memory_order_release);
    };
    publish(5);
    // Forces a completed row to reach CPU before later GPU-like work publishes.
    assert(first.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    for (size_t p : {3U, 1U, 6U, 2U, 4U, 0U}) publish(p);
    consumer.get();
    for (size_t p = 0; p < rows; ++p) {
        assert(seen[p] == 1);
        assert(std::equal(result[p].begin(), result[p].end(), mapped[p].begin()));
    }

    // Cancellation ends polling when a producer cannot publish remaining rows.
    cancelled.store(true);
    consume_ready(rows, 2, [](size_t) { assert(false); return false; },
        [](const auto&) { assert(false); }, cancelled);
    cancelled.store(false);
    unsigned batches = 0;
    consume_ready(rows, 2, [](size_t) { return true; }, [&](const auto&) {
        ++batches;
        cancelled.store(true, std::memory_order_release);
    }, cancelled);
    assert(batches == 1);

    // Consume must drain all partitions before an exception frees its scratch.
    cancelled.store(false);
    std::atomic<unsigned> drained{0};
    bool caught = false;
    try {
        consume_ready(rows, rows, [](size_t) { return true; }, [&](const auto& positions) {
            auto scratch = std::make_unique<std::vector<size_t>>(positions);
            pool.submit(positions.size(), [&](size_t i) {
                assert(scratch->at(i) == i);
                ++drained;
                if (i == 0) throw std::runtime_error("rerank failed");
            })->wait();
        }, cancelled);
    } catch (const std::runtime_error&) { caught = true; }
    assert(caught && drained == rows);
    pool.submit(1, [&](size_t) { ++drained; })->wait();
    assert(drained == rows + 1);

    const std::vector<uint8_t> coverage{0, 1, 0, 1};
    assert(!predicate_verified(nullptr, 1));
    assert(!predicate_verified(&coverage, 0));
    assert(predicate_verified(&coverage, 1));
    assert(!predicate_verified(&coverage, 99));
    bool zero_rejected = false;
    try { consume_ready(1, 0, [](size_t) { return true; }, [](const auto&) {}, cancelled); }
    catch (const std::invalid_argument&) { zero_rejected = true; }
    assert(zero_rejected);
    consume_ready(0, 1, [](size_t) { assert(false); return false; }, [](const auto&) {}, cancelled);
    std::cout << "PQ posting completion mapping, ordering, cancellation, drain/lifetime and coverage: PASS\n";
}
