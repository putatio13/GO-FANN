#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <thread>
#include <vector>

namespace go_fann::query_prepare {

inline size_t worker_count(uint32_t requested, size_t tasks) {
    const size_t available = std::max(1U, std::thread::hardware_concurrency());
    const size_t workers = requested ? requested : std::min<size_t>(8, available);
    return std::min(workers, tasks);
}

// Workers write only their task's output. Join before inspecting errors, and
// report the first input-order failure, independent of scheduling order.
template <class Function>
void parallel_tasks(size_t tasks, uint32_t requested, Function function) {
    const size_t workers = worker_count(requested, tasks);
    if (workers <= 1) {
        for (size_t task = 0; task < tasks; ++task) function(task);
        return;
    }
    std::atomic<size_t> next{0};
    std::vector<std::exception_ptr> errors(tasks);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    const auto join = [&] { for (auto& thread : threads) if (thread.joinable()) thread.join(); };
    try {
        for (size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&] {
                for (;;) {
                    const size_t task = next.fetch_add(1, std::memory_order_relaxed);
                    if (task >= tasks) return;
                    try { function(task); } catch (...) { errors[task] = std::current_exception(); }
                }
            });
        }
    } catch (...) {
        join();
        throw;
    }
    join();
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
}

}  // namespace go_fann::query_prepare
