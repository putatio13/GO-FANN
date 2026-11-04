#pragma once

#include "profiling/cpu_rerank_profile.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace go_fann::backend_rabitq {

struct RabitqCpuWorkerBudgetStats {
    size_t capacity = 0;
    size_t active_workers = 0;
    size_t peak_active_workers = 0;
    uint64_t acquisitions = 0;
    uint64_t contended_acquisitions = 0;
    uint64_t wait_ns = 0;
};

class RabitqCpuWorkerBudget {
public:
    class Permit {
    public:
        Permit() = default;
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;

        Permit(Permit&& other) noexcept : owner_(std::exchange(other.owner_, nullptr)) {}

        Permit& operator=(Permit&& other) noexcept
        {
            if (this != &other) {
                release();
                owner_ = std::exchange(other.owner_, nullptr);
            }
            return *this;
        }

        ~Permit() { release(); }

    private:
        friend class RabitqCpuWorkerBudget;
        explicit Permit(RabitqCpuWorkerBudget* owner) : owner_(owner) {}

        void release() noexcept
        {
            if (owner_ != nullptr) {
                owner_->release();
                owner_ = nullptr;
            }
        }

        RabitqCpuWorkerBudget* owner_ = nullptr;
    };

    explicit RabitqCpuWorkerBudget(size_t capacity) : capacity_(capacity)
    {
        if (capacity_ == 0) {
            throw std::invalid_argument("RabitQ CPU worker budget capacity must be positive");
        }
    }

    Permit acquire()
    {
        const auto wait_start = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        const bool contended = active_workers_ >= capacity_;
        condition_.wait(lock, [&] { return active_workers_ < capacity_; });
        const auto acquired_at = std::chrono::steady_clock::now();
        ++active_workers_;
        peak_active_workers_ = std::max(peak_active_workers_, active_workers_);
        ++acquisitions_;
        if (contended) {
            ++contended_acquisitions_;
            wait_ns_ += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    acquired_at - wait_start)
                    .count());
        }
        return Permit(this);
    }

    RabitqCpuWorkerBudgetStats snapshot_stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return RabitqCpuWorkerBudgetStats{capacity_,
                                         active_workers_,
                                         peak_active_workers_,
                                         acquisitions_,
                                         contended_acquisitions_,
                                         wait_ns_};
    }

private:
    void release() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --active_workers_;
        }
        condition_.notify_one();
    }

    const size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    size_t active_workers_ = 0;
    size_t peak_active_workers_ = 0;
    uint64_t acquisitions_ = 0;
    uint64_t contended_acquisitions_ = 0;
    uint64_t wait_ns_ = 0;
};

template <typename Fn>
decltype(auto) run_with_rabitq_cpu_worker_permit(RabitqCpuWorkerBudget* budget, Fn&& fn)
{
    if (budget == nullptr) {
        return std::forward<Fn>(fn)();
    }
    cpu_rerank_profile::Task profile("worker_permit", "coordinator");
    profile.stage(cpu_rerank_profile::Stage::WorkerWait);
    auto permit = budget->acquire();
    profile.stage(cpu_rerank_profile::Stage::Delegated);
    return std::forward<Fn>(fn)();
}

}  // namespace go_fann::backend_rabitq
