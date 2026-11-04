#pragma once

#include "execution/rabitq_cpu_affinity.h"

#include <condition_variable>
#include <cstdio>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace go_fann::backend_rabitq {

// Search-local CPU rerank workers shared by graph and posting consumers.
// Submit static partitions, not individual queries. GPU responders stay separate.
// Callers may submit concurrently; waiting for one batch never locks out another.
// Callbacks must not synchronously submit work to this same pool and wait for it.
class RabitqCpuWorkerPool {
public:
    class Batch {
    public:
        Batch(size_t partitions, std::function<void(size_t)> work)
            : partitions_(partitions), remaining_(partitions), work_(std::move(work)) {}

        void wait()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            done_.wait(lock, [&] { return remaining_ == 0; });
            if (error_) std::rethrow_exception(error_);
        }

    private:
        friend class RabitqCpuWorkerPool;

        void execute(size_t partition) noexcept
        {
            std::exception_ptr error;
            try {
                work_(partition);
            } catch (...) {
                error = std::current_exception();
            }
            bool complete;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (error && !error_) error_ = error;
                complete = --remaining_ == 0;
            }
            if (complete) done_.notify_all();
        }

        const size_t partitions_;
        size_t next_ = 0;  // Protected by the pool mutex.
        size_t remaining_;  // Protected by the batch mutex.
        std::function<void(size_t)> work_;
        std::mutex mutex_;
        std::condition_variable done_;
        std::exception_ptr error_;
    };

    explicit RabitqCpuWorkerPool(size_t capacity)
        : affinity_(std::getenv("GO_FANN_CPU_RERANK_CPUS"))
    {
        if (capacity == 0) throw std::invalid_argument("rerank worker capacity must be positive");
        workers_.reserve(capacity);
        try {
            for (size_t i = 0; i < capacity; ++i) {
                workers_.emplace_back([this] { worker_loop(); });
            }
            // All OS threads exist before setup completes, including a no-warmup run.
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [&] { return started_ == capacity; });
            if (startup_error_) std::rethrow_exception(startup_error_);
            if (affinity_.enabled()) {
                std::printf("[cpu-rerank-affinity] pool_workers=%zu cpus=%s verified=1\n",
                            capacity, affinity_.text().c_str());
            }
        } catch (...) {
            stop_and_join();
            throw;
        }
    }

    RabitqCpuWorkerPool(const RabitqCpuWorkerPool&) = delete;
    RabitqCpuWorkerPool& operator=(const RabitqCpuWorkerPool&) = delete;
    ~RabitqCpuWorkerPool() { stop_and_join(); }

    size_t capacity() const { return workers_.size(); }

    std::shared_ptr<Batch> submit(size_t partitions, std::function<void(size_t)> work)
    {
        auto batch = std::make_shared<Batch>(partitions, std::move(work));
        if (partitions == 0) return batch;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) throw std::logic_error("rerank worker pool is stopping");
            // Publish once: allocation failure cannot leave a partially queued batch.
            pending_.push_back(batch);
        }
        ready_.notify_all();
        return batch;
    }

private:
    void worker_loop()
    {
        std::exception_ptr error;
        try {
            affinity_.bind_current_thread();
        } catch (...) {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (error && !startup_error_) startup_error_ = error;
            ++started_;
        }
        ready_.notify_all();
        if (error) return;
        for (;;) {
            std::shared_ptr<Batch> batch;
            size_t partition;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                if (pending_.empty()) return;
                batch = pending_.front();
                partition = batch->next_++;
                if (batch->next_ == batch->partitions_) pending_.pop_front();
            }
            batch->execute(partition);
        }
    }

    void stop_and_join() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (auto& worker : workers_) if (worker.joinable()) worker.join();
    }

    RabitqCpuAffinity affinity_;
    std::exception_ptr startup_error_;
    std::vector<std::thread> workers_;
    std::deque<std::shared_ptr<Batch>> pending_;
    std::mutex mutex_;
    std::condition_variable ready_;
    size_t started_ = 0;
    bool stopping_ = false;
};

}  // namespace go_fann::backend_rabitq
