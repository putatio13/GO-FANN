#include "execution/rabitq_cpu_worker_pool.h"
#include "execution/rabitq_cpu_worker_budget.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <future>
#include <set>

using go_fann::backend_rabitq::RabitqCpuWorkerPool;
using go_fann::backend_rabitq::RabitqCpuWorkerBudget;
using go_fann::backend_rabitq::run_with_rabitq_cpu_worker_permit;
using namespace std::chrono_literals;

int main()
{
    // Startup affinity failures reach the creator and do not strand workers.
    assert(setenv("GO_FANN_CPU_RERANK_CPUS", "1023", 1) == 0);
    bool affinity_rejected = false;
    try { RabitqCpuWorkerPool invalid_affinity(3); }
    catch (const std::exception&) { affinity_rejected = true; }
    assert(affinity_rejected);
    assert(unsetenv("GO_FANN_CPU_RERANK_CPUS") == 0);

    for (const char* invalid : {"-1", "1-0", "1,", "1,,2", "1-", "1024", "9999999999999999"}) {
        affinity_rejected = false;
        try { go_fann::backend_rabitq::RabitqCpuAffinity affinity(invalid); }
        catch (const std::invalid_argument&) { affinity_rejected = true; }
        assert(affinity_rejected);
    }
    cpu_set_t inherited;
    assert(pthread_getaffinity_np(pthread_self(), sizeof(inherited), &inherited) == 0);
    int first_cpu = 0;
    while (!CPU_ISSET(first_cpu, &inherited)) ++first_cpu;
    const std::string cpu_text = std::to_string(first_cpu);
    assert(setenv("GO_FANN_CPU_RERANK_CPUS", cpu_text.c_str(), 1) == 0);
    {
        RabitqCpuWorkerPool pinned(2);
        std::atomic<int> arrived{0};
        pinned.submit(2, [&](size_t) {
            cpu_set_t actual;
            assert(pthread_getaffinity_np(pthread_self(), sizeof(actual), &actual) == 0);
            assert(CPU_COUNT(&actual) == 1 && CPU_ISSET(first_cpu, &actual));
            ++arrived;
            const auto deadline = std::chrono::steady_clock::now() + 5s;
            while (arrived != 2) {
                if (std::chrono::steady_clock::now() > deadline)
                    throw std::runtime_error("affinity startup test stalled");
                std::this_thread::yield();
            }
        })->wait();
    }
    assert(unsetenv("GO_FANN_CPU_RERANK_CPUS") == 0);
    cpu_set_t creator_mask;
    assert(pthread_getaffinity_np(pthread_self(), sizeof(creator_mask), &creator_mask) == 0);
    assert(CPU_EQUAL(&creator_mask, &inherited));

    bool rejected = false;
    try { RabitqCpuWorkerPool pool(0); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);

    RabitqCpuWorkerPool pool(3);
    assert(pool.capacity() == 3);
    pool.submit(0, [](size_t) { throw std::runtime_error("empty batch executed"); })->wait();
    std::set<std::thread::id> original_threads;
    for (int generation = 0; generation < 8; ++generation) {
        std::mutex mutex;
        std::condition_variable condition;
        size_t arrived = 0;
        std::set<std::thread::id> ids;
        std::vector<int> values(3, -1);
        auto batch = pool.submit(3, [&](size_t partition) {
            values[partition] = generation * 3 + static_cast<int>(partition);
            std::unique_lock<std::mutex> lock(mutex);
            ids.insert(std::this_thread::get_id());
            ++arrived;
            condition.notify_all();
            if (!condition.wait_for(lock, 5s, [&] { return arrived == 3; }))
                throw std::runtime_error("workers did not all run");
        });
        batch->wait();
        assert(ids.size() == 3);
        if (generation == 0) original_threads = ids;
        assert(ids == original_threads);
        for (int i = 0; i < 3; ++i) assert(values[i] == generation * 3 + i);
    }

    // A later provider can finish while an earlier provider still has blocked work.
    std::promise<void> release;
    auto released = release.get_future().share();
    std::promise<void> entered;
    auto first = pool.submit(1, [&](size_t) {
        entered.set_value();
        if (released.wait_for(5s) != std::future_status::ready)
            throw std::runtime_error("provider overlap stalled");
    });
    assert(entered.get_future().wait_for(5s) == std::future_status::ready);
    std::atomic<size_t> completed{0};
    auto second = std::async(std::launch::async, [&] {
        pool.submit(9, [&](size_t) { ++completed; })->wait();
    });
    const bool overlapped = second.wait_for(5s) == std::future_status::ready;
    release.set_value();
    first->wait();
    second.get();
    assert(overlapped && completed == 9);

    // An exception drains every partition, reaches the caller, and leaves reuse safe.
    completed = 0;
    auto failing = pool.submit(19, [&](size_t i) {
        ++completed;
        if (i == 4) throw std::runtime_error("expected worker error");
    });
    rejected = false;
    try { failing->wait(); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && completed == 19);
    pool.submit(1, [&](size_t) { ++completed; })->wait();
    assert(completed == 20);

    // Persistent posting and external graph workers obey the same permit budget.
    RabitqCpuWorkerBudget budget(2);
    std::atomic<size_t> active{0};
    auto score = [&](size_t) {
        run_with_rabitq_cpu_worker_permit(&budget, [&] {
            assert(++active <= 2);
            std::this_thread::sleep_for(200us);
            --active;
        });
    };
    auto posting = pool.submit(37, score);
    std::vector<std::thread> graph;
    for (size_t i = 0; i < 8; ++i) graph.emplace_back([&] { score(0); });
    posting->wait();
    for (auto& thread : graph) thread.join();
    auto stats = budget.snapshot_stats();
    assert(stats.acquisitions == 45 && stats.active_workers == 0);
    assert(stats.peak_active_workers == 2 && active == 0);

    // Graph and posting coordinators share one pool, retain separate scratch,
    // and wait only for their own batch before reusing it for the next generation.
    auto consume = [&](size_t width, int base) {
        std::vector<int> scratch(width, -1);
        for (int generation = 0; generation < 9; ++generation) {
            pool.submit(width, [&](size_t partition) {
                run_with_rabitq_cpu_worker_permit(&budget, [&] {
                    assert(++active <= 2);
                    scratch[partition] = base + generation * 100 + static_cast<int>(partition);
                    std::this_thread::yield();
                    --active;
                });
            })->wait();
            for (size_t i = 0; i < width; ++i)
                assert(scratch[i] == base + generation * 100 + static_cast<int>(i));
        }
    };
    auto pooled_graph = std::async(std::launch::async, [&] { consume(17, 1000); });
    auto pooled_posting = std::async(std::launch::async, [&] { consume(37, 9000); });
    assert(pooled_graph.wait_for(5s) == std::future_status::ready);
    assert(pooled_posting.wait_for(5s) == std::future_status::ready);
    pooled_graph.get();
    pooled_posting.get();
    stats = budget.snapshot_stats();
    assert(stats.acquisitions == 45 + 9 * (17 + 37));
    assert(stats.active_workers == 0 && stats.peak_active_workers == 2 && active == 0);

    // Shutdown drains accepted jobs before releasing the worker resources.
    completed = 0;
    std::shared_ptr<RabitqCpuWorkerPool::Batch> pending;
    {
        RabitqCpuWorkerPool draining(2);
        pending = draining.submit(31, [&](size_t) { ++completed; });
    }
    pending->wait();
    assert(completed == 31);
}
