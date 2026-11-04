#pragma once

#include "execution/rabitq_cpu_worker_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

namespace go_fann::posting_completion {

// Optional host scheduling control. Zero preserves synchronous callbacks.
inline size_t stream_inflight_from_env()
{
    const char* value = std::getenv("GO_FANN_POSTING_STREAM_INFLIGHT");
    if (value == nullptr || *value == '\0') return 0;
    size_t count = 0;
    for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || count > 8)
            throw std::invalid_argument("GO_FANN_POSTING_STREAM_INFLIGHT must be in [0,8]");
        count = count * 10 + static_cast<size_t>(*p - '0');
    }
    if (count > 8)
        throw std::invalid_argument("GO_FANN_POSTING_STREAM_INFLIGHT must be in [0,8]");
    return count;
}

// Finalizer coordinators stay outside the scoring pool: their callbacks may
// submit scoring partitions to that pool and wait. Slots own independent
// scratch; a slot cannot be reused before its previous callback has returned.
class BatchPipeline {
public:
    BatchPipeline(size_t inflight, std::atomic<bool>& cancelled)
        : cancelled_(cancelled), occupied_(inflight, false)
    {
        if (inflight == 0 || inflight > 8)
            throw std::invalid_argument("posting pipeline capacity must be in [1,8]");
        coordinators_ = std::make_unique<backend_rabitq::RabitqCpuWorkerPool>(inflight);
    }

    BatchPipeline(const BatchPipeline&) = delete;
    BatchPipeline& operator=(const BatchPipeline&) = delete;

    ~BatchPipeline()
    {
        drain();
        coordinators_.reset();
    }

    bool submit(std::function<void(size_t)> work)
    {
        size_t slot;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            while (active_ == occupied_.size() && !error_ && !cancelled_.load())
                ready_.wait_for(lock, std::chrono::milliseconds(1));
            if (error_) std::rethrow_exception(error_);
            if (cancelled_.load(std::memory_order_acquire)) return false;
            slot = 0;
            while (occupied_[slot]) ++slot;
            occupied_[slot] = true;
            ++active_;
            peak_ = std::max(peak_, active_);
        }
        try {
            coordinators_->submit(1, [this, slot, work = std::move(work)](size_t) {
                std::exception_ptr error;
                try {
                    if (!cancelled_.load(std::memory_order_acquire)) work(slot);
                } catch (...) {
                    error = std::current_exception();
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (error && !error_) {
                        error_ = error;
                        cancelled_.store(true, std::memory_order_release);
                    }
                    occupied_[slot] = false;
                    --active_;
                }
                ready_.notify_all();
            });
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                occupied_[slot] = false;
                --active_;
            }
            ready_.notify_all();
            throw;
        }
        return true;
    }

    void finish()
    {
        drain();
        if (error_) std::rethrow_exception(error_);
    }

    size_t peak() const { return peak_; } // Read only after finish/drain.

private:
    void drain() noexcept
    {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return active_ == 0; });
    }

    std::atomic<bool>& cancelled_;
    std::vector<bool> occupied_;
    std::unique_ptr<backend_rabitq::RabitqCpuWorkerPool> coordinators_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::exception_ptr error_;
    size_t active_ = 0;
    size_t peak_ = 0;
};

} // namespace go_fann::posting_completion
