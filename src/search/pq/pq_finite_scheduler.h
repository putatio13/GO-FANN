#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "execution/async_io.h"

namespace go_fann::backend_pq {

enum class PqFiniteQueryIdError {
    NONE,
    TOO_MANY,
    OUT_OF_RANGE,
    DUPLICATE,
};

struct PqFiniteQueryIdValidation {
    PqFiniteQueryIdError error = PqFiniteQueryIdError::NONE;
    size_t position = 0;
    uint32_t query_id = 0;
};

inline PqFiniteQueryIdValidation validate_finite_query_ids(
    const std::vector<uint32_t>& query_ids,
    size_t query_capacity)
{
    if (query_ids.size() > query_capacity) {
        return {PqFiniteQueryIdError::TOO_MANY, query_ids.size(), 0};
    }
    std::vector<uint8_t> seen(query_capacity, 0);
    for (size_t position = 0; position < query_ids.size(); ++position) {
        const uint32_t query_id = query_ids[position];
        if (query_id >= query_capacity) {
            return {PqFiniteQueryIdError::OUT_OF_RANGE, position, query_id};
        }
        if (seen[query_id] != 0) {
            return {PqFiniteQueryIdError::DUPLICATE, position, query_id};
        }
        seen[query_id] = 1;
    }
    return {};
}

inline bool finite_mailbox_identity_is_valid(int64_t row,
                                             int64_t buffer,
                                             int64_t generation)
{
    return row >= 0 && static_cast<uint64_t>(row) <= UINT32_MAX &&
           buffer >= 0 && buffer <= 1 &&
           generation > 0 && static_cast<uint64_t>(generation) <= UINT32_MAX;
}

inline bool finite_ready_batch_should_flush(size_t batch_size,
                                            size_t batch_capacity,
                                            uint64_t oldest_ready_age_ns,
                                            uint64_t timeout_ns)
{
    if (batch_size == 0 || batch_capacity == 0) {
        return false;
    }
    return batch_size >= batch_capacity || oldest_ready_age_ns >= timeout_ns;
}

inline uint64_t finite_partial_batch_remaining_wait_ns(uint64_t oldest_published_ns,
                                                       uint64_t now_ns,
                                                       uint64_t timeout_ns)
{
    if (now_ns <= oldest_published_ns) {
        return timeout_ns;
    }
    const uint64_t age_ns = now_ns - oldest_published_ns;
    return age_ns >= timeout_ns ? 0 : timeout_ns - age_ns;
}

inline uint32_t next_finite_persistent_sequence(uint32_t current)
{
    if (current >= UINT32_MAX - 1) {
        throw std::overflow_error("finite persistent worker sequence exhausted");
    }
    return current + 1;
}

inline uint64_t finite_ready_ring_required_consumed_sequence(uint64_t ticket,
                                                             uint32_t ring_capacity)
{
    if (ring_capacity == 0) {
        throw std::invalid_argument("finite ready ring capacity must be positive");
    }
    return ticket < ring_capacity ? 0 : ticket - ring_capacity + 1;
}

constexpr uint64_t kPqFiniteReadyRingDoneBit = 1ULL << 32;
constexpr uint64_t kPqFiniteReadyRingPublishingBit = 1ULL << 33;
constexpr uint64_t kPqFiniteReadyRingGenerationMask = UINT32_MAX;

constexpr uint32_t finite_ready_ring_query_generation(uint64_t state) noexcept
{
    return static_cast<uint32_t>(state & kPqFiniteReadyRingGenerationMask);
}

constexpr bool finite_ready_ring_query_done(uint64_t state) noexcept
{
    return (state & kPqFiniteReadyRingDoneBit) != 0;
}

constexpr bool finite_ready_ring_query_publishing(uint64_t state) noexcept
{
    return (state & kPqFiniteReadyRingPublishingBit) != 0;
}

enum class PqFiniteReadyRingClaimResult {
    CLAIMED,
    DUPLICATE,
    TERMINAL,
    PUBLISHING,
};

inline PqFiniteReadyRingClaimResult finite_ready_ring_try_claim_ready(
    std::atomic<uint64_t>& query_state,
    uint32_t generation) noexcept
{
    uint64_t prior_state = query_state.load(std::memory_order_acquire);
    while (true) {
        if (finite_ready_ring_query_done(prior_state)) {
            return PqFiniteReadyRingClaimResult::TERMINAL;
        }
        if (generation <= finite_ready_ring_query_generation(prior_state)) {
            return PqFiniteReadyRingClaimResult::DUPLICATE;
        }
        if (finite_ready_ring_query_publishing(prior_state)) {
            return PqFiniteReadyRingClaimResult::PUBLISHING;
        }
        const uint64_t next_state = static_cast<uint64_t>(generation) |
                                    kPqFiniteReadyRingPublishingBit;
        if (query_state.compare_exchange_weak(prior_state,
                                              next_state,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
            return PqFiniteReadyRingClaimResult::CLAIMED;
        }
    }
}

inline void finite_ready_ring_finish_ready(std::atomic<uint64_t>& query_state,
                                           uint32_t generation) noexcept
{
    query_state.store(static_cast<uint64_t>(generation), std::memory_order_release);
}

inline PqFiniteReadyRingClaimResult finite_ready_ring_try_claim_done(
    std::atomic<uint64_t>& query_state) noexcept
{
    uint64_t prior_state = query_state.load(std::memory_order_acquire);
    while (true) {
        if (finite_ready_ring_query_done(prior_state)) {
            return PqFiniteReadyRingClaimResult::TERMINAL;
        }
        if (finite_ready_ring_query_publishing(prior_state)) {
            return PqFiniteReadyRingClaimResult::PUBLISHING;
        }
        if (query_state.compare_exchange_weak(
                prior_state,
                prior_state | kPqFiniteReadyRingDoneBit,
                std::memory_order_acq_rel,
                std::memory_order_acquire)) {
            return PqFiniteReadyRingClaimResult::CLAIMED;
        }
    }
}

struct PqFiniteCompletionQueueStats {
    uint64_t ready_enqueued = 0;
    uint64_t done_enqueued = 0;
    uint64_t dequeued = 0;
    uint64_t duplicate_rejections = 0;
    uint64_t invalid_rejections = 0;
    uint64_t terminal_rejections = 0;
    uint64_t cancelled_rejections = 0;
    uint64_t backpressure_waits = 0;
    size_t max_depth = 0;
};

class PqFiniteCompletionQueue final : public HostCompletionSink {
public:
    explicit PqFiniteCompletionQueue(const std::vector<uint32_t>& active_query_ids)
        : capacity_(active_query_ids.size()),
          ring_(capacity_),
          query_states_(capacity_)
    {
        query_slots_.reserve(active_query_ids.size());
        for (size_t slot = 0; slot < active_query_ids.size(); ++slot) {
            if (!query_slots_.emplace(active_query_ids[slot], slot).second) {
                throw std::invalid_argument("finite completion queue query IDs must be unique");
            }
        }
    }

    ~PqFiniteCompletionQueue() override
    {
        cancel();
    }

    PqFiniteCompletionQueue(const PqFiniteCompletionQueue&) = delete;
    PqFiniteCompletionQueue& operator=(const PqFiniteCompletionQueue&) = delete;

    bool publish_ready(uint32_t query_id,
                       uint32_t generation,
                       uint32_t row,
                       uint32_t buffer) override
    {
        const HostCompletionEvent event{HostCompletionKind::Ready,
                                        query_id,
                                        generation,
                                        row,
                                        buffer,
                                        steady_now_ns()};
        return publish_ready_batch(&event, 1);
    }

    bool publish_ready_batch(const HostCompletionEvent* events, size_t count) override
    {
        if (events == nullptr && count != 0) return false;
        if (count == 0) return true;
        std::unique_lock<std::mutex> lock(mutex_);
        if (cancelled_) {
            ++stats_.cancelled_rejections;
            return false;
        }
        if (count > capacity_) {
            ++stats_.invalid_rejections;
            return false;
        }

        std::unordered_map<uint32_t, size_t> batch_slots;
        batch_slots.reserve(count);
        for (size_t idx = 0; idx < count; ++idx) {
            const HostCompletionEvent& event = events[idx];
            const auto slot_it = query_slots_.find(event.query_id);
            if (event.kind != HostCompletionKind::Ready ||
                slot_it == query_slots_.end() || event.generation == 0 ||
                event.row == UINT32_MAX || event.buffer > 1) {
                ++stats_.invalid_rejections;
                return false;
            }
            if (!batch_slots.emplace(event.query_id, slot_it->second).second) {
                ++stats_.duplicate_rejections;
                return false;
            }
            const QueryState& state = query_states_[slot_it->second];
            if (state.done) {
                ++stats_.terminal_rejections;
                return false;
            }
            if (state.has_ready && event.generation <= state.last_generation) {
                ++stats_.duplicate_rejections;
                return false;
            }
        }

        wait_for_capacity(lock, count);
        if (cancelled_) {
            ++stats_.cancelled_rejections;
            return false;
        }

        // wait_for_capacity releases the mutex. Recheck per-query ownership
        // before committing the all-or-nothing batch after it reacquires it.
        for (size_t idx = 0; idx < count; ++idx) {
            const HostCompletionEvent& event = events[idx];
            const QueryState& state = query_states_[batch_slots.at(event.query_id)];
            if (state.done) {
                ++stats_.terminal_rejections;
                return false;
            }
            if (state.has_ready && event.generation <= state.last_generation) {
                ++stats_.duplicate_rejections;
                return false;
            }
        }

        for (size_t idx = 0; idx < count; ++idx) {
            const HostCompletionEvent& event = events[idx];
            QueryState& state = query_states_[batch_slots.at(event.query_id)];
            state.has_ready = true;
            state.last_generation = event.generation;
            enqueue_locked(event, false);
        }
        stats_.ready_enqueued += count;
        not_empty_.notify_one();
        return true;
    }

    bool publish_done(uint32_t query_id) override
    {
        std::unique_lock<std::mutex> lock(mutex_);
        const auto slot_it = query_slots_.find(query_id);
        if (slot_it == query_slots_.end()) {
            ++stats_.invalid_rejections;
            return false;
        }
        QueryState& state = query_states_[slot_it->second];
        if (cancelled_) {
            ++stats_.cancelled_rejections;
            return false;
        }
        if (state.done) {
            ++stats_.duplicate_rejections;
            return false;
        }
        wait_for_capacity(lock, 1);
        if (cancelled_) {
            ++stats_.cancelled_rejections;
            return false;
        }
        if (state.done) {
            ++stats_.duplicate_rejections;
            return false;
        }

        state.done = true;
        enqueue_locked({HostCompletionKind::Done,
                        query_id,
                        0,
                        UINT32_MAX,
                        0,
                        steady_now_ns()});
        ++stats_.done_enqueued;
        return true;
    }

    bool wait_pop_batch(std::vector<HostCompletionEvent>& out,
                        size_t max_batch_size,
                        std::chrono::nanoseconds partial_timeout)
    {
        if (max_batch_size == 0) {
            throw std::invalid_argument("finite completion batch size must be positive");
        }
        out.clear();
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock,
                                 partial_timeout,
                                 [this] { return cancelled_ || size_ != 0; })) {
            return true;
        }
        if (cancelled_) return false;

        const uint64_t remaining_wait_ns = finite_partial_batch_remaining_wait_ns(
            ring_[head_].published_ns,
            steady_now_ns(),
            static_cast<uint64_t>(partial_timeout.count()));
        const auto deadline = std::chrono::steady_clock::now() +
                              std::chrono::nanoseconds(remaining_wait_ns);
        while (!cancelled_ && size_ < max_batch_size) {
            if (not_empty_.wait_until(lock, deadline) == std::cv_status::timeout) break;
        }
        if (cancelled_) return false;

        const size_t count = std::min(size_, max_batch_size);
        out.reserve(std::max(out.capacity(), count));
        for (size_t i = 0; i < count; ++i) {
            out.push_back(ring_[head_]);
            head_ = (head_ + 1) % capacity_;
        }
        size_ -= count;
        stats_.dequeued += count;
        lock.unlock();
        not_full_.notify_all();
        return true;
    }

    bool try_pop_batch(std::vector<HostCompletionEvent>& out, size_t max_batch_size)
    {
        if (max_batch_size == 0) {
            throw std::invalid_argument("finite completion batch size must be positive");
        }
        out.clear();
        std::unique_lock<std::mutex> lock(mutex_);
        if (cancelled_) return false;
        const size_t count = std::min(size_, max_batch_size);
        out.reserve(std::max(out.capacity(), count));
        for (size_t i = 0; i < count; ++i) {
            out.push_back(ring_[head_]);
            head_ = (head_ + 1) % capacity_;
        }
        size_ -= count;
        stats_.dequeued += count;
        lock.unlock();
        if (count != 0) not_full_.notify_all();
        return true;
    }

    void cancel() noexcept override
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cancelled_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t capacity() const noexcept
    {
        return capacity_;
    }

    PqFiniteCompletionQueueStats stats() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

private:
    struct QueryState {
        uint32_t last_generation = 0;
        bool has_ready = false;
        bool done = false;
    };

    static uint64_t steady_now_ns()
    {
        return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }

    void wait_for_capacity(std::unique_lock<std::mutex>& lock, size_t count)
    {
        if (count > capacity_) {
            throw std::invalid_argument("finite completion batch exceeds queue capacity");
        }
        if (size_ > capacity_ - count && !cancelled_) ++stats_.backpressure_waits;
        not_full_.wait(lock, [this, count] { return cancelled_ || size_ <= capacity_ - count; });
    }

    void enqueue_locked(const HostCompletionEvent& event, bool notify = true)
    {
        ring_[tail_] = event;
        tail_ = (tail_ + 1) % capacity_;
        ++size_;
        stats_.max_depth = std::max(stats_.max_depth, size_);
        if (notify) not_empty_.notify_one();
    }

    const size_t capacity_;
    std::vector<HostCompletionEvent> ring_;
    std::unordered_map<uint32_t, size_t> query_slots_;
    std::vector<QueryState> query_states_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t size_ = 0;
    bool cancelled_ = false;
    PqFiniteCompletionQueueStats stats_{};
};

}  // namespace go_fann::backend_pq
