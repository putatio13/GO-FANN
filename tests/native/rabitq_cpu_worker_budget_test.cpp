#include "execution/rabitq_cpu_worker_budget.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

void update_max(std::atomic<size_t>& maximum, size_t value)
{
    size_t observed = maximum.load(std::memory_order_relaxed);
    while (observed < value &&
           !maximum.compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
    }
}

}  // namespace

int main()
{
    using go_fann::backend_rabitq::RabitqCpuWorkerBudget;
    using go_fann::backend_rabitq::run_with_rabitq_cpu_worker_permit;

    RabitqCpuWorkerBudget budget(2);
    std::atomic<size_t> active{0};
    std::atomic<size_t> maximum{0};
    std::vector<std::thread> workers;
    for (size_t worker = 0; worker < 8; ++worker) {
        workers.emplace_back([&] {
            run_with_rabitq_cpu_worker_permit(&budget, [&] {
                const size_t now = active.fetch_add(1, std::memory_order_relaxed) + 1;
                update_max(maximum, now);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                active.fetch_sub(1, std::memory_order_relaxed);
            });
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    const auto saturated = budget.snapshot_stats();
    assert(maximum.load(std::memory_order_relaxed) <= 2);
    assert(saturated.capacity == 2);
    assert(saturated.active_workers == 0);
    assert(saturated.peak_active_workers == 2);
    assert(saturated.acquisitions == 8);
    assert(saturated.contended_acquisitions > 0);

    bool threw = false;
    try {
        run_with_rabitq_cpu_worker_permit(&budget, [] {
            throw std::runtime_error("expected");
        });
    } catch (const std::runtime_error&) {
        threw = true;
    }
    assert(threw);
    assert(budget.snapshot_stats().active_workers == 0);
    return 0;
}
